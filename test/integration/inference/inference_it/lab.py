"""The lab under test: forwards, gateways, keys, oracle, and the traffic-case primitive."""
from __future__ import annotations

import http.client
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from .cluster import Cluster, Forward, OcError, PodForwards
from .config import (INPUT_SERIES, KEY_NAMES, LABEL_MODEL, LABEL_TENANT, LABEL_USER,
                     LABEL_WORKLOAD, OUTPUT_SERIES, SCRAPE_JOBS, Config)
from .gateway import ChatResult, Gateway
from .metrics import Row, delta_rows
from .opencost import OpenCost
from .oracle import EngineOracle, OracleDelta, diff
from .prom import Prom

import loadgen  # noqa: E402  (tools/inference-loadgen; config, imported above, sets the path)


class LabNotReady(RuntimeError):
    pass


@dataclass
class Outcome:
    """Everything one traffic case observed, from the three read channels."""
    alias: str
    results: list[ChatResult]
    oracle: OracleDelta
    rows: list[Row]
    client_in: int
    client_out: int
    t0: float
    t1: float

    @property
    def e_in(self) -> float:
        return self.oracle.prompt

    @property
    def e_out(self) -> float:
        return self.oracle.generation

    @property
    def statuses(self) -> list:
        return [r.status for r in self.results]

    def attributed(self, **labels) -> tuple[float, float]:
        """Sum LiteLLM counter deltas whose labels contain every given label value."""
        i = o = 0.0
        for r in self.rows:
            if all(r.labels.get(k) == v for k, v in labels.items()):
                if r.metric == INPUT_SERIES:
                    i += r.delta
                else:
                    o += r.delta
        return i, o

    def series(self) -> list[tuple]:
        return sorted({(r.labels.get(LABEL_TENANT), r.labels.get(LABEL_WORKLOAD),
                        r.labels.get(LABEL_USER), r.labels.get(LABEL_MODEL)) for r in self.rows},
                      key=str)

    def total_attributed(self) -> tuple[float, float]:
        return self.attributed()


class Lab:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cluster = Cluster(cfg.oc)
        self.engines: dict[str, PodForwards] = {}
        self.litellm_pods: PodForwards | None = None
        self.prom: Forward | None = None
        self.master_key = ""
        self._oc_forward: Forward | None = None
        self._oc_api: OpenCost | None = None
        self._keys: dict[str, str] = {}
        self._created: list[tuple[str, str]] = []   # (team_id, key) made by the harness
        self.notes: list[str] = []
        self.store: dict = {}                        # cross-case data (O-19 totals for O-20/O-21)

    # --- lifecycle -------------------------------------------------------------------------
    def prepare(self, wait_s: int = 0) -> None:
        """Forward every engine pod and LiteLLM pod. Raises LabNotReady with the reason."""
        c, cfg = self.cluster, self.cfg
        want = {"gemma-4-31B": 2, "gemma-4-31B-b": 1, "llama-chat": 1}
        groups = cfg.engine_groups()
        missing = []
        for alias, g in groups.items():
            pods = c.wait_pods(g.namespace, g.selector, want[alias], wait_s) if wait_s else \
                c.pods(g.namespace, g.selector)
            if len(pods) < want[alias]:
                missing.append(f"{alias}: {len(pods)}/{want[alias]} pods ready in {g.namespace}")
        if missing:
            raise LabNotReady("engines not ready: " + "; ".join(missing))
        for alias, g in groups.items():
            self.engines[alias] = PodForwards(c, g.namespace, g.selector, g.port)
            self.engines[alias].sync()
        self.litellm_pods = PodForwards(c, cfg.litellm_ns, cfg.litellm_selector, 4000)
        if len(self.litellm_pods.sync()) < 1:
            raise LabNotReady("no ready LiteLLM pod")
        self.master_key = c.secret(cfg.litellm_ns, "litellm-secrets", "LITELLM_MASTER_KEY")
        self.prom = Forward(cfg.oc, cfg.oc_ns, f"svc/{cfg.prometheus_svc}", cfg.prometheus_port).start()
        if not self.prom.wait_open():
            raise LabNotReady(f"Prometheus {cfg.oc_ns}/{cfg.prometheus_svc} not reachable")

    def shutdown(self) -> None:
        for team_id, key in self._created:
            try:
                self.gw().admin("/key/delete", {"keys": [key]})
                self.gw().admin("/team/delete", {"team_ids": [team_id]})
            except Exception:   # best-effort cleanup of harness-made identities
                pass
        for pf in [*self.engines.values(), self.litellm_pods]:
            if pf:
                pf.stop()
        if self.prom:
            self.prom.stop()
        if self._oc_forward:
            self._oc_forward.stop()

    # --- accessors -------------------------------------------------------------------------
    def gateways(self) -> list[Gateway]:
        assert self.litellm_pods
        return [Gateway(fw.port, self.master_key, name) for name, fw in
                sorted(self.litellm_pods.forwards.items())]

    def gw(self, index: int = 0) -> Gateway:
        """One LiteLLM replica. Counters are per replica, so a case sends its traffic and reads
        its counters through the same one (plan G-16 treats the two-replica case separately)."""
        return self.gateways()[index]

    def key(self, name: str) -> str:
        if name not in KEY_NAMES:
            raise KeyError(name)
        if name not in self._keys:
            self._keys[name] = self.cluster.secret(self.cfg.oc_ns, "litellm-virtual-keys", name)
        return self._keys[name]

    def oracle(self, *aliases: str) -> EngineOracle:
        return EngineOracle(*[self.engines[a] for a in aliases])

    # --- Prometheus and OpenCost (Phase 2) ---------------------------------------------------
    def promq(self) -> Prom:
        assert self.prom, "prepare() was not called"
        return Prom(self.prom.port)

    def opencost(self) -> OpenCost:
        """OpenCost API behind a loopback forward, started on first use so Phase 0 and 1 runs do
        not need it. The admin token is read from the cluster secret and kept in memory."""
        if self._oc_api is None:
            cfg = self.cfg
            self._oc_forward = Forward(cfg.oc, cfg.oc_ns, f"svc/{cfg.opencost_svc}",
                                       cfg.opencost_port).start()
            if not self._oc_forward.wait_open():
                raise LabNotReady(f"OpenCost service {cfg.oc_ns}/{cfg.opencost_svc} not reachable")
            token = self.cluster.secret(cfg.oc_ns, cfg.admin_secret, "ADMIN_TOKEN")
            self._oc_api = OpenCost(self._oc_forward.port, token)
        return self._oc_api

    def mark(self) -> float:
        """Now, in Prometheus's clock. Scrape timestamps and `time()` use it, not the local clock."""
        return self.promq().server_time()

    def wait_scrapes(self, since: float, scrapes: int = 2, timeout_s: float | None = None) -> dict:
        """Wait until every healthy LiteLLM and engine target has been scraped at least `scrapes`
        times after `since` (a `mark()` taken after the last request).

        Plan Section 5.1: `increase()` omits a series with a single sample and OpenCost never
        widens its window, so a case may only query once two scrapes follow its last request.
        Returns {"scrape_interval_s", "waited_s", "counts", "ok"}; `ok` is False on timeout.
        """
        prom = self.promq()
        interval = prom.scrape_interval(SCRAPE_JOBS)
        timeout = timeout_s if timeout_s is not None else interval * (scrapes + 3) + 30
        targets = prom.active_targets(SCRAPE_JOBS)
        started = time.monotonic()
        counts: dict = {}
        while True:
            counts = prom.scrape_counts(since, SCRAPE_JOBS)
            done = bool(targets) and all(counts.get(t, 0) >= scrapes for t in targets)
            if done or time.monotonic() - started >= timeout:
                return {"scrape_interval_s": interval, "waited_s": round(time.monotonic() - started, 1),
                        "targets": len(targets), "counts": {f"{j}/{i}": n for (j, i), n in counts.items()},
                        "ok": done}
            time.sleep(min(5.0, max(1.0, interval / 6)))

    def engine_chat(self, alias: str, prompt: str, out_tokens: int = 8,
                    timeout: float = 120) -> ChatResult:
        """Send one completion DIRECTLY to an engine pod, bypassing LiteLLM (plan O-04).

        Uses any ready pod of the alias's group. vLLM-only length forcing is dropped on HTTP 400,
        like traffic_resilient does for gateway traffic; the oracle, not the forced length, is
        the authority for E_out.
        """
        g = self.cfg.engine_groups()[alias]
        fw = self.engines[alias].forwards[sorted(self.engines[alias].forwards)[0]]
        body = {"model": g.served_model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": out_tokens,
                "min_tokens": out_tokens, "ignore_eos": True}

        def post(b: dict) -> ChatResult:
            conn = http.client.HTTPConnection("127.0.0.1", fw.port, timeout=timeout)
            try:
                conn.request("POST", "/v1/chat/completions", json.dumps(b),
                             {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
            finally:
                conn.close()
            try:
                payload = json.loads(data or b"{}")
            except json.JSONDecodeError:
                payload = {}
            usage = payload.get("usage") or {}
            err = None if resp.status == 200 else str(payload.get("error", payload))[:300]
            return ChatResult(resp.status, usage.get("prompt_tokens"),
                              usage.get("completion_tokens"), err, body=payload)

        res = post(body)
        if res.status == 400:
            for f in ("min_tokens", "ignore_eos"):
                body.pop(f, None)
            res = post(body)
        return res

    def oc_logs(self, tail: int = 500) -> str:
        """Recent logs of the OpenCost deployment (over-attribution and unmatched-key warnings)."""
        cfg = self.cfg
        return self.cluster.run("logs", f"deployment/{cfg.opencost_deploy}", "-n", cfg.oc_ns,
                                f"--tail={tail}", timeout=90)

    def restart_opencost(self, timeout: int = 300) -> None:
        """Rollout-restart OpenCost and wait for one ready pod. Raises OcError on failure."""
        cfg, c = self.cfg, self.cluster
        c.run("rollout", "restart", f"deployment/{cfg.opencost_deploy}", "-n", cfg.oc_ns)
        c.run("rollout", "status", f"deployment/{cfg.opencost_deploy}", "-n", cfg.oc_ns,
              f"--timeout={timeout}s", timeout=timeout + 30)
        c.wait_pods(cfg.oc_ns, cfg.opencost_selector, 1, 180)
        if self._oc_forward:
            # The API tunnel was bound to the OLD pod (oc port-forward svc/... pins one pod).
            # Bounce it so the supervisor loop re-attaches to the new pod before a case queries.
            self._oc_forward.bounce()
            if not self._oc_forward.wait_open(timeout=60):
                raise LabNotReady("OpenCost API forward did not re-open after the restart")

    def attribution_config_text(self) -> str:
        """The current attribution.yaml from the cluster (the source of truth, never a local file)."""
        cfg = self.cfg
        # Escape dots in the key: jsonpath reads `.data.attribution.yaml` as data -> attribution
        # -> yaml (absent, empty string). The key is `attribution.yaml`.
        key = cfg.attribution_key.replace(".", "\\.")
        return self.cluster.run("get", "configmap", cfg.attribution_configmap, "-n", cfg.oc_ns,
                                "-o", f"jsonpath={{.data.{key}}}")

    @contextmanager
    def attribution_config(self, text: str):
        """Run a block with a modified attribution.yaml, then restore the original whatever happens.

        Patches the ConfigMap, restarts OpenCost (the collector reads the mapping at startup) and
        waits for the rollout. The original text is restored in `finally`, so a failed case cannot
        leave the lab on a variant configuration.
        """
        cfg, c = self.cfg, self.cluster
        original = self.attribution_config_text()
        patch = json.dumps({"data": {cfg.attribution_key: text}})
        try:
            c.run("patch", "configmap", cfg.attribution_configmap, "-n", cfg.oc_ns,
                  "--type", "merge", "-p", patch)
            self.restart_opencost()
            yield
        finally:
            c.run("patch", "configmap", cfg.attribution_configmap, "-n", cfg.oc_ns,
                  "--type", "merge", "-p", json.dumps({"data": {cfg.attribution_key: original}}))
            self.restart_opencost()

    @staticmethod
    def set_mapping_label(text: str, label: str, value: str) -> str:
        """Set one `label: value` line of the producer mapping (V3: userLabel off, V7: workloadLabel
        off). Raises ValueError when the label does not occur exactly once, so a config format
        change is a BLOCKED case, not a silent no-op."""
        changed, n = re.subn(rf"(?m)^(\s*{re.escape(label)}:\s*).*$", rf'\1"{value}"', text)
        if n != 1:
            raise ValueError(f"attribution config: expected one {label} line, found {n}")
        return changed

    def _deploy_env(self) -> dict[str, str]:
        """Plain NAME=value environment of the OpenCost deployment's first container."""
        out = {}
        for line in self.cluster.run("set", "env", f"deployment/{self.cfg.opencost_deploy}",
                                     "-n", self.cfg.oc_ns, "--list").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip().isidentifier():
                    out[k.strip()] = v.strip()
        return out

    @contextmanager
    def opencost_env(self, **kv: str):
        """Run a block with OpenCost deployment environment variables overridden, then restore.

        Used by the Phase 3 cases to toggle INFERENCE_ATTRIBUTION_ENABLED (A-03) and
        INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED (A-04). The original values are restored
        in `finally`; a variable that was unset before is removed again with `NAME-`.
        """
        cfg, c = self.cfg, self.cluster
        before = self._deploy_env()
        try:
            c.run("set", "env", f"deployment/{cfg.opencost_deploy}", "-n", cfg.oc_ns,
                  *[f"{k}={v}" for k, v in kv.items()])
            self.restart_opencost()
            yield
        finally:
            args: list[str] = []
            for k in kv:
                if k in before:
                    args.append(f"{k}={before[k]}")
                else:
                    args.append(f"{k}-")     # oc set env NAME- removes the variable
            c.run("set", "env", f"deployment/{cfg.opencost_deploy}", "-n", cfg.oc_ns, *args)
            self.restart_opencost()

    def new_identity(self, team_alias: str, key_alias: str | None) -> str:
        """Create a team and a key owned by the harness. Removed in shutdown()."""
        gw = self.gw()
        team_id = gw.admin("/team/new", {"team_alias": team_alias})["team_id"]
        body = {"team_id": team_id}
        if key_alias:
            body["key_alias"] = key_alias
        key = gw.admin("/key/generate", body)["key"]
        self._created.append((team_id, key))
        return key

    def versions(self) -> dict:
        v = {}
        try:
            lp = self.cluster.pods(self.cfg.litellm_ns, self.cfg.litellm_selector)
            if lp:
                v["litellm image"] = lp[0]["image"]
                v["litellm digest"] = lp[0]["imageID"]
                v["litellm replicas"] = str(len(lp))
            for alias, pf in self.engines.items():
                pods = list(pf.pods.values())
                if pods:
                    v[f"engine {alias}"] = f"{pods[0]['image']} {pods[0]['imageID']}"
        except OcError as e:
            v["version lookup"] = f"failed: {e}"
        return v

    # --- traffic primitive -----------------------------------------------------------------
    def observe(self, alias: str, action: Callable[[], list[ChatResult]], *,
                oracle_aliases: tuple[str, ...] | None = None,
                gw: Gateway | list[Gateway] | None = None, settle_s: float = 6) -> Outcome:
        """Run `action` between two reads of the engine oracle and the gateway counters.

        The oracle waits until the engines are idle on both sides, so aborted or timed-out
        requests that the engine is still finishing are included in the delta. `gw` may be a
        list (one per LiteLLM replica): each replica's counters are differenced on their own and
        the rows concatenated, which is how a per-replica `increase()` is summed.
        """
        gws = gw if isinstance(gw, list) else [gw or self.gw()]
        oracle = self.oracle(*(oracle_aliases if oracle_aliases is not None else (alias,)))
        e0 = oracle.wait_idle(quiet_s=4)
        c0 = [g.wait_counters_stable() for g in gws]
        t0 = time.time()
        results = action()
        e1 = oracle.wait_idle(quiet_s=settle_s)
        c1 = [g.wait_counters_stable() for g in gws]
        t1 = time.time()
        rows = [r for a, b in zip(c0, c1) for r in delta_rows(a, b)]
        ok = [r for r in results if r.ok]
        return Outcome(alias, results, diff(e0, e1), rows,
                       sum(r.prompt_tokens or 0 for r in ok), sum(r.completion_tokens or 0 for r in ok),
                       t0, t1)

    def traffic(self, alias: str, key_name_or_key: str, *, n: int = 3, prompt_tokens: int = 200,
                out_tokens: int = 16, user: str | None = None, stream: bool = False,
                oracle_aliases: tuple[str, ...] | None = None, gw: Gateway | None = None,
                content: str | None = None, headers: dict | None = None, extra: dict | None = None,
                include_usage: bool = True, force_length: bool = True,
                abort_after_chunks: int | None = None, settle_s: float = 6) -> Outcome:
        gw = gw or self.gw()
        key = self.key(key_name_or_key) if key_name_or_key in KEY_NAMES else key_name_or_key
        prompt = content if content is not None else loadgen.build_prompt(prompt_tokens)

        def action() -> list[ChatResult]:
            out = []
            for _ in range(n):
                body = gw.chat_body(alias, prompt, out_tokens, user, stream, force_length, extra,
                                    include_usage)
                out.append(gw.chat(key, body, headers, abort_after_chunks=abort_after_chunks))
            return out

        return self.observe(alias, action, oracle_aliases=oracle_aliases, gw=gw, settle_s=settle_s)
