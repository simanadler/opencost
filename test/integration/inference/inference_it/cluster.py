"""`oc` wrapper, secret access and loopback port-forwards.

The lab cluster API is intermittently slow (TLS handshake timeouts), and a port-forward can
accept connections yet stop relaying. So `oc` calls retry on transient errors, and every
forward runs in a supervisor loop that restarts it and can be bounced on demand.

All forwards bind to 127.0.0.1 only. Secrets are read at run time and kept in memory.
"""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import time

TRANSIENT = ("TLS handshake timeout", "i/o timeout", "connection refused", "EOF",
             "Unable to connect", "context deadline exceeded", "etcdserver")


class OcError(RuntimeError):
    pass


class Cluster:
    def __init__(self, oc: str = "oc"):
        self.oc = oc
        self._secrets: dict[tuple[str, str], dict] = {}

    def run(self, *args: str, timeout: int = 60, retries: int = 4) -> str:
        last = ""
        for _ in range(retries):
            try:
                p = subprocess.run([self.oc, *args], capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                last = "timeout"
                time.sleep(2)
                continue
            if p.returncode == 0:
                return p.stdout
            last = p.stderr.strip()
            if not any(t in last for t in TRANSIENT):
                break
            time.sleep(2)
        raise OcError(f"oc {' '.join(args[:3])} failed: {last[-200:]}")

    def get_json(self, *args: str) -> dict:
        return json.loads(self.run("get", *args, "-o", "json"))

    def secret(self, ns: str, name: str, key: str) -> str:
        if (ns, name) not in self._secrets:
            self._secrets[(ns, name)] = self.get_json("secret", name, "-n", ns).get("data", {})
        data = self._secrets[(ns, name)]
        if key not in data:
            raise OcError(f"secret {ns}/{name} has no key {key!r}")
        return base64.b64decode(data[key]).decode()

    def pods(self, ns: str, selector: str) -> list[dict]:
        """Running, ready, not-terminating pods with name, image and image digest."""
        out = []
        for item in self.get_json("pods", "-n", ns, "-l", selector).get("items", []):
            meta, status = item["metadata"], item.get("status", {})
            cs = status.get("containerStatuses") or []
            if meta.get("deletionTimestamp") or status.get("phase") != "Running":
                continue
            if not cs or not all(c.get("ready") for c in cs):
                continue
            out.append({"name": meta["name"], "image": cs[0].get("image", ""),
                        "imageID": cs[0].get("imageID", ""),
                        "restarts": sum(c.get("restartCount", 0) for c in cs)})
        return sorted(out, key=lambda p: p["name"])

    def wait_pods(self, ns: str, selector: str, count: int, timeout: int) -> list[dict]:
        deadline = time.monotonic() + timeout
        pods: list[dict] = []
        while time.monotonic() < deadline:
            try:
                pods = self.pods(ns, selector)
            except OcError:
                pods = []
            if len(pods) >= count:
                return pods
            time.sleep(10)
        return pods

    def delete_pod(self, ns: str, name: str, force: bool = False) -> None:
        args = ["delete", "pod", name, "-n", ns, "--wait=false"]
        if force:
            args += ["--grace-period=0", "--force"]
        self.run(*args)

    def logs(self, ns: str, pod: str, tail: int = 500) -> str:
        return self.run("logs", pod, "-n", ns, f"--tail={tail}", timeout=60)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Forward:
    """`oc port-forward <target> <local>:<remote>` under a restart loop, loopback only."""

    def __init__(self, oc: str, ns: str, target: str, remote_port: int):
        self.oc, self.ns, self.target, self.remote_port = oc, ns, target, remote_port
        self.port = _free_port()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Forward":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._proc = subprocess.Popen(
                [self.oc, "port-forward", self.target, f"{self.port}:{self.remote_port}",
                 "-n", self.ns, "--address", "127.0.0.1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not self._stop.is_set() and self._proc.poll() is None:
                time.sleep(0.5)
            if self._proc.poll() is None:
                self._proc.kill()
            time.sleep(1)

    def bounce(self) -> None:
        """Kill the current tunnel; the supervisor loop starts a fresh one."""
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def stop(self) -> None:
        self._stop.set()
        self.bounce()

    def wait_open(self, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=2):
                    return True
            except OSError:
                time.sleep(0.5)
        return False


class PodForwards:
    """One forward per ready pod matching a selector. `sync()` follows pod changes."""

    def __init__(self, cluster: Cluster, ns: str, selector: str, remote_port: int):
        self.cluster, self.ns, self.selector, self.remote_port = cluster, ns, selector, remote_port
        self.forwards: dict[str, Forward] = {}
        self.pods: dict[str, dict] = {}

    def sync(self) -> list[str]:
        pods = {p["name"]: p for p in self.cluster.pods(self.ns, self.selector)}
        for name in list(self.forwards):
            if name not in pods:
                self.forwards.pop(name).stop()
        for name in pods:
            if name not in self.forwards:
                self.forwards[name] = Forward(self.cluster.oc, self.ns, f"pod/{name}",
                                              self.remote_port).start()
        self.pods = pods
        for fw in self.forwards.values():
            fw.wait_open()
        return sorted(self.forwards)

    def stop(self) -> None:
        for fw in self.forwards.values():
            fw.stop()
        self.forwards.clear()
