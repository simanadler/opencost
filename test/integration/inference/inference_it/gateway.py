"""LiteLLM client: chat (plain, streaming, streaming with early disconnect), admin API, metrics."""
from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import INPUT_SERIES, OUTPUT_SERIES
from .metrics import Series, parse, to_series


@dataclass
class ChatResult:
    status: int | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    latency_s: float = 0.0
    content_chunks: int = 0
    usage_chunk_empty_choices: bool | None = None   # G-02: extra usage-only chunk?
    aborted: bool = False                           # client closed the stream early
    body: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == 200


class Gateway:
    """One LiteLLM pod (via its loopback forward) with the master key held in memory only."""

    def __init__(self, port: int, master_key: str, name: str = "litellm"):
        self.port, self.master_key, self.name = port, master_key, name

    # --- low level -------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None, token: str | None = None,
                 headers: dict | None = None, timeout: float = 120) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        h.update(headers or {})
        try:
            conn.request(method, path, json.dumps(body) if body is not None else None, h)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def get(self, path: str, token: str | None = None, timeout: float = 30) -> tuple[int, str]:
        status, data = self._request("GET", path, token=token, timeout=timeout)
        return status, data.decode(errors="replace")

    def admin(self, path: str, body: dict) -> dict:
        status, data = self._request("POST", path, body, self.master_key, timeout=60)
        if status != 200:
            raise RuntimeError(f"admin {path} -> {status}: {data[:300]!r}")
        return json.loads(data or b"{}")

    # --- chat ------------------------------------------------------------------------------
    @staticmethod
    def chat_body(model: str, content: str, max_tokens: int, user: str | None, stream: bool,
                  force_length: bool = True, extra: dict | None = None,
                  include_usage: bool = True) -> dict:
        body: dict = {"model": model, "messages": [{"role": "user", "content": content}],
                      "temperature": 0, "stream": stream}
        if max_tokens:
            body["max_tokens"] = max_tokens
            if force_length:           # vLLM extensions: exact completion length (plan 5.1)
                body["min_tokens"] = max_tokens
                body["ignore_eos"] = True
        if user is not None:
            body["user"] = user
        if stream and include_usage:
            body["stream_options"] = {"include_usage": True}
        body.update(extra or {})
        return body

    def chat(self, key: str, body: dict, headers: dict | None = None, timeout: float = 180,
             abort_after_chunks: int | None = None,
             on_chunk: Callable[[int], None] | None = None) -> ChatResult:
        start = time.monotonic()
        if body.get("stream"):
            return self._chat_stream(key, body, headers, timeout, abort_after_chunks, start, on_chunk)
        try:
            status, data = self._request("POST", "/v1/chat/completions", body, key, headers, timeout)
        except (OSError, http.client.HTTPException) as e:
            return ChatResult(None, error=str(e)[:200], latency_s=time.monotonic() - start)
        try:
            payload = json.loads(data or b"{}")
        except json.JSONDecodeError:
            payload = {}
        usage = payload.get("usage") or {}
        err = None if status == 200 else json.dumps(payload.get("error", payload))[:300]
        return ChatResult(status, usage.get("prompt_tokens"), usage.get("completion_tokens"), err,
                          time.monotonic() - start, body=payload)

    def _chat_stream(self, key, body, headers, timeout, abort_after, start, on_chunk=None) -> ChatResult:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        h.update(headers or {})
        res = ChatResult(None)
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body), h)
            resp = conn.getresponse()
            res.status = resp.status
            if resp.status != 200:
                res.error = resp.read().decode(errors="replace")[:300]
                return res
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    res.prompt_tokens = obj["usage"].get("prompt_tokens")
                    res.completion_tokens = obj["usage"].get("completion_tokens")
                    res.usage_chunk_empty_choices = not obj.get("choices")
                if obj.get("choices"):
                    res.content_chunks += 1
                    if on_chunk:
                        on_chunk(res.content_chunks)
                    if abort_after and res.content_chunks >= abort_after:
                        res.aborted = True
                        break
        except (OSError, http.client.HTTPException) as e:
            res.error = str(e)[:200]
        finally:
            conn.close()
            res.latency_s = time.monotonic() - start
        return res

    # --- metrics ---------------------------------------------------------------------------
    def metrics_text(self, authenticated: bool = True, path: str = "/metrics/") -> tuple[int, str]:
        return self.get(path, self.master_key if authenticated else None)

    def counters(self) -> Series:
        for attempt in range(5):
            try:
                status, text = self.metrics_text()
            except (OSError, http.client.HTTPException):
                time.sleep(2)
                continue
            if status == 200:
                return to_series(parse(text, {INPUT_SERIES, OUTPUT_SERIES}))
            time.sleep(2)
        raise RuntimeError(f"{self.name}: /metrics/ unreadable")

    def wait_counters_stable(self, stable_reads: int = 3, interval: float = 1.0,
                             timeout: float = 45) -> Series:
        """LiteLLM updates counters in an async callback; wait until they stop moving."""
        deadline = time.monotonic() + timeout
        last, same = self.counters(), 0
        while time.monotonic() < deadline and same < stable_reads:
            time.sleep(interval)
            cur = self.counters()
            same = same + 1 if cur == last else 0
            last = cur
        return last
