"""Run configuration. Names follow the OpenShift install guide; every value can be overridden
through the same environment variables the guide exports."""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# tools/inference-loadgen is reused for the fixed prompt corpus (plan Section 5.1).
_LOADGEN_DIR = Path(__file__).resolve().parents[4] / "tools" / "inference-loadgen"
if str(_LOADGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_LOADGEN_DIR))

# Sentinel LiteLLM emits for a missing team, key alias or user. Discovered by the first
# lab run (P-02 census, G-03) and pinned here; G-03 re-asserts it on every run.
SENTINEL = "None"

# Labels on the LiteLLM token counters (P-08). Phase 0 asserts these exist.
LABEL_TENANT = "team_alias"
LABEL_WORKLOAD = "api_key_alias"
LABEL_USER = "end_user"
LABEL_MODEL = "model"
LABEL_REQUESTED_MODEL = "requested_model"

INPUT_SERIES = "litellm_input_tokens_metric_total"
OUTPUT_SERIES = "litellm_output_tokens_metric_total"

# Prometheus scrape jobs whose samples Phase 2 waits for (install guide Step 2.8).
SCRAPE_JOBS = ("litellm", "inference-engines")

# Virtual keys created by install guide Step 2.7 and stored in the litellm-virtual-keys secret.
KEY_NAMES = ("rag", "batch", "engineering-rag", "chat", "agent", "noteam-key", "limited")

# Key name -> (team alias = tenant, key alias = workload), plan Section 4.1. None = no value.
MATRIX = {
    "rag": ("acme-corp/engineering", "rag"),
    "batch": ("acme-corp/engineering", "batch"),
    "engineering-rag": ("acme-corp", "engineering/rag"),
    "chat": ("globex", "chat"),
    "agent": ("globex", "agent"),
    "limited": ("limited", "limited"),
    "noteam-key": (None, None),
}


@dataclass
class EngineGroup:
    """The engine pods that can serve one client model alias."""
    alias: str
    namespace: str
    selector: str
    port: int
    served_model: str          # the engine's model_name label
    prefix: str = "vllm"       # metric prefix: vllm:* or sglang:*


@dataclass
class Config:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    oc_ns: str = os.environ.get("OC_NS", "opencost2")
    litellm_ns: str = os.environ.get("LITELLM_NS", "litellm-4cost2")
    engine_ns_a: str = os.environ.get("ENGINE_NS_A", "llm-a-4cost2")
    engine_ns_b: str = os.environ.get("ENGINE_NS_B", "llm-b-4cost2")
    prometheus_svc: str = os.environ.get("PROMETHEUS_SVC", "prometheus-lab-server")
    prometheus_port: int = int(os.environ.get("PROMETHEUS_PORT", "80"))
    opencost_svc: str = os.environ.get("OPENCOST_SVC", "opencost2")
    opencost_port: int = int(os.environ.get("OPENCOST_PORT", "9003"))
    opencost_deploy: str = os.environ.get("OPENCOST_DEPLOY", "opencost2")
    opencost_selector: str = os.environ.get("OPENCOST_SELECTOR", "app.kubernetes.io/name=opencost")
    admin_secret: str = os.environ.get("OPENCOST_ADMIN_SECRET", "opencost-admin")
    attribution_configmap: str = os.environ.get("OPENCOST_ATTRIBUTION_CONFIGMAP", "opencost-attribution")
    attribution_key: str = "attribution.yaml"
    scrape_interval_s: int = int(os.environ.get("PROMETHEUS_SCRAPE_INTERVAL", "30"))
    collection_interval_s: int = int(os.environ.get("INFERENCE_COLLECTION_INTERVAL_S", "60"))
    litellm_selector: str = "app=litellm"
    oc: str = os.environ.get("OC_BIN", "oc")
    out_dir: Path = Path(os.environ.get("INFERENCE_IT_OUT", str(Path(__file__).resolve().parents[1] / "runs")))
    recover_timeout_s: int = 1500

    # Plan Section 8 O-01: token coverage tolerance, 0.1 over and 0.9 under.
    coverage_over: float = 1.1
    coverage_under: float = 0.9

    def engine_groups(self) -> dict[str, EngineGroup]:
        return {
            "gemma-4-31B": EngineGroup("gemma-4-31B", self.engine_ns_a, "app=vllm-gemma", 8000, "gemma-4-31B"),
            "gemma-4-31B-b": EngineGroup("gemma-4-31B-b", self.engine_ns_b, "app=vllm-gemma", 8000, "gemma-4-31B"),
            "llama-chat": EngineGroup("llama-chat", self.engine_ns_a, "app=sglang-llama", 30000, "llama", "sglang"),
        }
