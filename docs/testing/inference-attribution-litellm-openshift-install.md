# Installation Guide: Inference Attribution Integration Test on OpenShift

Companion to [Integration Test Plan: Inference Cost Attribution with LiteLLM as the Gateway](./inference-attribution-litellm-integration-test-plan.md).

The installation is split into four phases, in dependency order. Each phase ends with a sanity test that proves it worked. Do not start a phase until the previous phase's sanity test passes.

| Phase | Components | Namespace(s) | Sanity test |
|---|---|---|---|
| 1 | vLLM (`gemma-4-31B`), SGLang (`llama`), and the Prometheus that scrapes them | `llm-a-4cost2`, `llm-b-4cost2`, `opencost2` | Step 1.7 |
| 2 | SaaS stub, LiteLLM gateway, PostgreSQL, virtual keys | `litellm-4cost2` | Step 2.9 |
| 3 | OpenCost (current branch, custom image) | `opencost2` | Step 3.5 |
| 4 | Load generator (client side) | Your workstation | Step 4.3 |

The phases must run in order. Phase 1 installs Prometheus with the engine scrape job. Phase 2 adds the LiteLLM job to that Prometheus. OpenCost is installed last, in Phase 3, so its health check sees the real token counters from the engines and LiteLLM.

The cluster already runs an OpenCost install in the `opencost` namespace. This guide never touches that namespace. Everything here lives in `opencost2`, `litellm-4cost2`, `llm-a-4cost2` and `llm-b-4cost2`.

---

## 0. Conventions and prerequisites

### 0.1 Conventions

- All commands run from the `opencost/` repository root on the `inference-v2-implementation` branch unless stated.
- Shell variables used throughout. Export them in every new shell. Phase 2 generates `LITELLM_MASTER_KEY` in the same shell, so keep that shell open across Phases 2 and 4:

```bash
export OC_NS=opencost2
export LITELLM_NS=litellm-4cost2
export ENGINE_NS_A=llm-a-4cost2
export ENGINE_NS_B=llm-b-4cost2
export GHCR_USERNAME=simanadler             # same account as BUILD_PUSH_DEPLOY.sh
export REGISTRY=ghcr.io/${GHCR_USERNAME}    # registry used by the repo's build scripts
export OPENCOST_IMAGE=${REGISTRY}/opencost-inference:inference-v2
```

- **No secrets in files or in git.** Every key, token and password is generated at install time and stored in an OpenShift `Secret`. Add the local values directory (Step 0.4) to `.gitignore` if you keep it inside the repo.
- **Port-forwards bind to `127.0.0.1` only** (`--address 127.0.0.1`). Never use `0.0.0.0`.

### 0.2 Cluster and tooling

| Requirement | Check |
|---|---|
| OpenShift 4.14 or later, with rights to create namespaces and cluster-scoped read objects | `oc whoami`, `oc auth can-i create namespace` |
| GPU worker nodes with the NVIDIA GPU Operator installed and healthy | `oc get nodes -l nvidia.com/gpu.present=true`, `oc get clusterpolicy` shows `ready` |
| Node Feature Discovery operator installed | `oc get nodefeaturediscovery -A` |
| `oc` CLI matching the cluster version | `oc version` |
| `helm` 3.14 or later | `helm version` |
| `docker` with `buildx` (or `podman` 4.9 or later) | `docker buildx version` |
| GitHub Container Registry (`ghcr.io`) access with a token that has `write:packages` and `read:packages` | `echo $GITHUB_TOKEN \| docker login ghcr.io -u ${GHCR_USERNAME} --password-stdin` |
| `jq`, `curl`, `openssl`, Python 3.11 or later on your workstation | `jq --version && python3 --version` |
| Hugging Face token with access to the gated model weights (Phase 1 only) | Kept in a Secret only |

Log in:

```bash
oc login --server=https://api.<cluster-domain>:6443
```

### 0.3 Confirm the OpenCost branch carries the required fixes

The plan requires #4161 (SGLang engine queries) and #4163 (model cost keyed by namespace). A search of the local commit history for these numbers found no matches, so confirm them before you build the OpenCost image in Step 3.1:

```bash
git branch --show-current                       # expect inference-v2-implementation
git log --oneline --grep='4161' --grep='4163' --grep='4162'
```

If the commits are not present, incorporate them into the branch as described in [Appendix A](#appendix-a-incorporating-the-opencost-prs-into-the-branch) before Step 3.1. Phase 0 check P-01 fails otherwise.

Record the build commit:

```bash
git rev-parse HEAD > ../opencost-build-commit.txt
```

### 0.4 Create the local values directory

The values files in later steps contain environment-specific names. Keep them outside git (in the directory above the `opencost/` repository):

```bash
mkdir -p ../inference-attribution-lab/runs
```

### 0.5 Check the cluster for name conflicts

The cluster already has an OpenCost install in `opencost`. Before Phase 1, confirm the names this guide uses are free:

```bash
oc get project opencost2 litellm-4cost2 llm-a-4cost2 llm-b-4cost2 2>&1 | grep -v NotFound
helm list -n opencost2          # expect no prometheus-lab release
```

The first command prints nothing when the projects do not exist yet. Confirm the `opencost` namespace is untouched: `helm list -n opencost` still shows the `opencost` and `prometheus-opencost` releases.

---

# Phase 1: engines and Prometheus

Deploys the real engines and the Prometheus that scrapes them. The engines are the oracle for token counts (plan Section 5.2). Prometheus in this phase scrapes only the engines. Phase 2 adds LiteLLM, and Phase 3 adds OpenCost, which reads from this Prometheus.

## 1.1 Create the namespaces and label the engine namespaces

```bash
oc new-project ${OC_NS} || oc project ${OC_NS}
oc new-project ${ENGINE_NS_A} || oc project ${ENGINE_NS_A}
oc new-project ${ENGINE_NS_B} || oc project ${ENGINE_NS_B}
```

Label the engine namespaces so the engine pods are found by the Prometheus job in Step 1.6:

```bash
oc label namespace ${ENGINE_NS_A} inference-lab=engines --overwrite
oc label namespace ${ENGINE_NS_B} inference-lab=engines --overwrite
```

## 1.2 Hugging Face secret

```bash
for ns in ${ENGINE_NS_A} ${ENGINE_NS_B}; do
  oc create secret generic hf-token -n ${ns} --from-literal=HF_TOKEN="<your-hf-token>"
done
```

Read the token from a shell prompt (`read -s HF_TOKEN`) rather than typing it inline.

## 1.3 Deploy vLLM (`gemma-4-31B`) in `llm-a-4cost2` and `llm-b-4cost2`

Pin the image to vLLM 0.31.0, the version measured in plan Section 2. Save as `../inference-attribution-lab/vllm-gemma.yaml`. Substitute `__NAMESPACE__` and `REPLICAS` with the `sed` commands below:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gemma
  namespace: __NAMESPACE__
  labels: { app: vllm-gemma, inference-lab: engine }
spec:
  replicas: REPLICAS
  progressDeadlineSeconds: 1800   # a 31B model load takes 7+ minutes; the 600 s default marks a healthy rollout as failed
  selector:
    matchLabels: { app: vllm-gemma }
  template:
    metadata:
      labels: { app: vllm-gemma, inference-lab: engine, llm-d.ai/model: gemma-4-31B }
    spec:
      containers:
        - name: vllm
          image: docker.io/vllm/vllm-openai:v0.31.0   # vllm/vllm-openai v0.31.0 (plan Section 2); see image note below
          args:
            - --model=google/gemma-4-31B-it
            - --served-model-name=gemma-4-31B        # A7: must match LiteLLM model_name suffix
            - --tensor-parallel-size=2
            - --port=8000
            - --enforce-eager
            - --max-model-len=8192
            - '--limit-mm-per-prompt={"image": 0, "audio": 0}'
          envFrom:
            - secretRef: { name: hf-token }
          env:
            - name: HOME
              value: /tmp
            - name: HF_HOME
              value: /tmp/hf-home
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 2
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 60
            periodSeconds: 10
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            - name: hf-cache
              mountPath: /tmp/hf-home
      volumes:
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: 16Gi }
        - name: hf-cache
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gemma
  namespace: __NAMESPACE__
spec:
  selector: { app: vllm-gemma }
  ports:
    - port: 8000
      targetPort: 8000
```

Apply to both namespaces, with the right replica count:

```bash
sed -e 's/__NAMESPACE__/llm-a-4cost2/g' -e 's/REPLICAS/2/' ../inference-attribution-lab/vllm-gemma.yaml | oc apply -n llm-a-4cost2 -f -
sed -e 's/__NAMESPACE__/llm-b-4cost2/g' -e 's/REPLICAS/1/' ../inference-attribution-lab/vllm-gemma.yaml | oc apply -n llm-b-4cost2 -f -
```

Each engine pod must carry the label `inference-lab: engine`. It must also carry `llm-d.ai/model`, set to exactly the value of the engine's `--served-model-name` (`gemma-4-31B` for vLLM, `llama` for SGLang). OpenCost joins pod cost to engine metrics through this label (the default `INFERENCE_MODEL_LABEL`). Without it every engine pod is filed under `__unallocated__` and the cost for the model is 0. The label only reaches OpenCost if kube-state-metrics exports it, which Step 1.6.1 configures. Confirm with `oc get pods -n llm-a-4cost2 --show-labels`.

> **Apply the manifest once, complete.** The `HOME` and `HF_HOME` variables are required: without them the Hugging Face client tries to write `/.cache` as the arbitrary OpenShift UID and the pod crash-loops with `PermissionError at /.cache`. If you apply an earlier version of the manifest and then re-apply a corrected one, the Deployment keeps overlapping ReplicaSets, and the old one can crash-loop while holding GPUs. Check with `oc get rs -n llm-a-4cost2`. Every ReplicaSet except the newest must show `DESIRED 0`. The new pods need a GPU pair each, so a rolling update replaces them one at a time and the old pod stays until the new one is Ready. The load takes 7 to 10 minutes per pod, so allow about 20 minutes for the two-replica rollout.
>
> A pod that is `Running` but not `Ready` for several minutes is loading weights, not failing. Confirm with `oc logs <pod> | tail`, which should end in `Loading model from scratch` or `Multi-modal warmup completed`, and that GPU memory is in use.
>
> Re-applying with a changed Deployment can leave a stale pod that never becomes Ready, because it was started from the old spec. Delete that pod and the ReplicaSet recreates it from the current spec.
>
> **GPU reaper.** This cluster runs a controller that scales a GPU Deployment to 0 after 240 minutes without traffic. It records `gpu-reaper.io/reason` and `gpu-reaper.io/scaled-down-at` on the Deployment, so `oc get deploy -n llm-a-4cost2 -o yaml | grep gpu-reaper` tells you it happened. Scaling to 0 deletes the pods, and because the model cache is an `emptyDir`, scaling back up re-downloads the weights (20 to 60 minutes for gemma, depending on Hugging Face throughput). A pod sitting at `Running` and not `Ready` that long is downloading, not hung: `oc exec <pod> -- du -sm /tmp/hf-home` should keep growing. To avoid the loss, send a request to each engine at least every few hours while the lab is idle, or ask the cluster admins to exempt the four lab namespaces. If it already happened, restore the replicas: `oc scale deploy/vllm-gemma -n llm-a-4cost2 --replicas=2`, `oc scale deploy/sglang-llama -n llm-a-4cost2 --replicas=1`, `oc scale deploy/vllm-gemma -n llm-b-4cost2 --replicas=1`. The SGLang manifest has no readiness probe, so its pod shows `Ready` as soon as the container starts; confirm the model has loaded by sending a completion (Step 1.7, Test 2) rather than trusting `READY 1/1`.

> **Image and model notes.** `vllm/vllm-openai` is an upstream Docker Hub image. Mirror it into the lab's ghcr.io namespace, `${REGISTRY}/vllm-openai:v0.31.0`, and reference the mirror, so every image the lab runs comes from ghcr.io. Record the mirror source and digest in the run report, and record the deviation from `registry.redhat.io` with the reason. Replace `<hf-repo-for-gemma-4-31B>` with the Hugging Face repository the lab uses. The plan lists `gemma-4-31B` and `gemma-4-31B-it` as candidates, so confirm the exact name.

## 1.4 Deploy SGLang (`llama`) in `llm-a-4cost2`

Save as `../inference-attribution-lab/sglang-llama.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-llama
  namespace: llm-a-4cost2
  labels: { app: sglang-llama, inference-lab: engine }
spec:
  replicas: 1
  selector:
    matchLabels: { app: sglang-llama }
  template:
    metadata:
      labels: { app: sglang-llama, inference-lab: engine, llm-d.ai/model: llama }
    spec:
      containers:
        - name: sglang
          image: docker.io/lmsysorg/sglang:v0.5.22   # lmsysorg/sglang:v0.5.22; see image note below
          command: ["python3", "-m", "sglang.launch_server"]
          args:
            - --model-path=unsloth/Meta-Llama-3.1-8B-Instruct
            - --served-model-name=llama
            - --host=0.0.0.0
            - --port=30000
            - --enable-metrics
          envFrom:
            - secretRef: { name: hf-token }
          env:
            - name: HOME
              value: /tmp
            - name: HF_HOME
              value: /tmp/hf-home
          ports:
            - containerPort: 30000
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            - name: hf-cache
              mountPath: /tmp/hf-home
      volumes:
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: 8Gi }
        - name: hf-cache
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-llama
  namespace: llm-a-4cost2
spec:
  selector: { app: sglang-llama }
  ports:
    - name: http
      port: 30000
      targetPort: 30000
```

> **Security deviation to record.** `--host=0.0.0.0` binds the engine to all pod interfaces, which the project rules forbid. It is needed here so the Service can reach the engine. Restrict ingress with a NetworkPolicy that allows port 30000 only from the `litellm-4cost2` namespace, and record the exception in the run report. The vLLM Deployment in Step 1.3 has the same default bind and needs the same NetworkPolicy.
>
> Two things to confirm for SGLang. First, its metrics port: SGLang serves `/metrics` on its API port (30000 here) when `--enable-metrics` is set, not on 8000 like vLLM. The Prometheus job in Step 1.6 scrapes each pod's declared container port, so no per-engine port setting is needed, provided every engine Deployment declares its `containerPort`. Confirm with `oc exec deploy/sglang-llama -n llm-a-4cost2 -- python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:30000/metrics').read().decode()[:300])"`. The counters carry `engine_type="unified"` and `model_name="llama"`. Second, plan Section 2 says SGLang reports `sglang:prompt_tokens_total` and that prefill/decode disaggregation needs `engine_type` filtering (O-14). Start with a single non-disaggregated worker.

```bash
oc apply -f ../inference-attribution-lab/sglang-llama.yaml
oc rollout status deployment/sglang-llama -n llm-a-4cost2 --timeout=20m
```

## 1.5 Port-forward the engine oracle channel

Port-forward one engine pod's metrics to your workstation, bound to loopback. This is the oracle channel in plan Section 3:

```bash
POD=$(oc get pods -n ${ENGINE_NS_A} -l app=vllm-gemma -o jsonpath='{.items[0].metadata.name}')
oc port-forward pod/${POD} 18000:8000 -n ${ENGINE_NS_A} --address 127.0.0.1 &
curl -s http://127.0.0.1:18000/metrics | grep -E '^vllm:(prompt|generation)_tokens_total' | head
```

Expect the counters with `model_name="gemma-4-31B"`. Record the exact label set.

## 1.6 Install Prometheus with the engine scrape job

The cluster's existing Prometheus release `prometheus-opencost` in the `opencost` namespace stays untouched. The lab gets a separate release, `prometheus-lab`, with a different name so `helm list -A` and `oc get svc -A` stay unambiguous.

Use the community chart. The repository's `kubernetes/prometheus/extraScrapeConfigs.yaml` targets the `opencost` namespace, so do not use it unchanged.

### 1.6.1 Create the values file

Save as `../inference-attribution-lab/prometheus-values.yaml`. This Phase 1 file scrapes only the engines. Phase 2 adds the LiteLLM job.

Three settings here are not optional, because OpenCost's cost and token joins depend on them. Each failure is silent:

- **`namespace` relabel on the engine job.** Engine series carry no `namespace` label. OpenCost files them under `namespace="unknown"`, so tokens never meet the cost entry for the real namespace.
- **`metricLabelsAllowlist` on kube-state-metrics.** v2 exports no pod labels by default, so `kube_pod_labels` does not exist and every pod lands in `__unallocated__`. The model cost is then 0.
- **Scrape-job scoping, `retention` and `retentionSize`.** The chart's default jobs scrape the whole cluster. On a shared cluster they filled a 20Gi volume in about 3 days. Prometheus then stopped with `write to WAL: disk quota exceeded` and OpenCost crash-looped with `no running jobs on Prometheus`.

The scoped jobs keep other teams' workloads out of Prometheus. Note that kube-state-metrics still reports every pod in the cluster. Other teams' llm-d models can therefore appear in `/inferenceCost/total`, with cost but no tokens, and their cost lands in the `__unmetered__` tenant. The attribution config's `scope.servingModels` limits the attribution side.

```yaml
extraScrapeConfigs: |
  # Engine counters (vllm:* and sglang:*). honor_labels keeps model_name intact (P-04).
  # Scrapes each pod's declared container port (vLLM 8000, SGLang 30000); pod-role discovery
  # already puts that port in __address__.
  - job_name: inference-engines
    honor_labels: true
    scrape_interval: 30s
    metrics_path: /metrics
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [llm-a-4cost2, llm-b-4cost2]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_inference_lab]
        regex: engine
        action: keep
      # OpenCost joins engine tokens to model cost on model_name:namespace, so the engine
      # series must carry the pod's namespace. Without it they are filed under
      # namespace="unknown" and never match the cost entry for the real namespace.
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

# Replace the chart's default cluster-wide scrape jobs. The lab only needs the engine job
# above (Phase 2 adds the LiteLLM job), plus the cadvisor container metrics and the
# kube-state-metrics that OpenCost's allocation queries read, limited to the lab namespaces.
# The default cluster-wide scrape filled the 20Gi volume in about 3 days and stopped
# Prometheus (write to WAL: disk quota exceeded).
scrapeConfigs:
  kubernetes-api-servers:
    enabled: false
  kubernetes-nodes:
    enabled: false
  kubernetes-service-endpoints:
    enabled: false
  kubernetes-service-endpoints-slow:
    enabled: false
  prometheus-pushgateway:
    enabled: false
  kubernetes-services:
    enabled: false
  kubernetes-pods:
    enabled: false
  kubernetes-pods-slow:
    enabled: false
  # Keep Prometheus scraping itself (the chart's own default job, unchanged).
  prometheus:
    enabled: true
  # cadvisor is scraped per node, so keep only series from the lab namespaces at ingest.
  kubernetes-nodes-cadvisor:
    enabled: true
    scheme: https
    metrics_path: /metrics/cadvisor
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      insecure_skip_verify: true
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - source_labels: [__metrics_path__]
        target_label: metrics_path
    metric_relabel_configs:
      - source_labels: [namespace]
        regex: (llm-a-4cost2|llm-b-4cost2)
        action: keep
  # kube-state-metrics from this release, for node, pod and PVC info.
  kube-state-metrics:
    enabled: true
    honor_labels: true
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: [opencost2]
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        regex: prometheus-lab-kube-state-metrics
        action: keep
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        regex: http
        action: keep

# OpenShift restricted-v2 SCC: let the platform assign UID/GID/fsGroup from the
# namespace range instead of the chart's hardcoded 65534.
server:
  securityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null
    runAsNonRoot: true
  persistentVolume:
    size: 20Gi
  # Bound the TSDB so it cannot fill the volume again.
  retention: 3d
  retentionSize: 12GB
kube-state-metrics:
  securityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null
    runAsNonRoot: true
  # kube-state-metrics v2 exports no pod labels unless allowlisted. OpenCost reads pod labels
  # from kube_pod_labels to group cost by model; without this every pod lands in
  # __unallocated__ and the model cost is $0. The label values must equal the engine's
  # model_name (--served-model-name).
  metricLabelsAllowlist:
    - pods=[llm-d.ai/model,llm-d.ai/inference-shared]
# node-exporter needs hostNetwork/hostPID/hostPath, which no SCC usable here allows.
prometheus-node-exporter:
  enabled: false
```

> **Why the OpenShift block is required.** The chart hardcodes UID, GID and fsGroup 65534. OpenShift's `restricted-v2` SCC only admits IDs from the namespace's assigned range, so without the block the server and kube-state-metrics ReplicaSets fail with `unable to validate against any security context constraint` and `oc rollout status` times out with no pods created. Setting the IDs to `null` lets OpenShift assign them. `runAsNonRoot` stays true.

Confirm the `extraScrapeConfigs` key against the chart you install:

```bash
helm show values prometheus-community/prometheus | grep -n extraScrapeConfigs
```

### 1.6.2 Install

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm list -n ${OC_NS}          # expect no prometheus-lab release
helm install prometheus-lab prometheus-community/prometheus \
  -n ${OC_NS} \
  -f ../inference-attribution-lab/prometheus-values.yaml \
  --set alertmanager.enabled=false \
  --set kube-state-metrics.enabled=true \
  --set prometheus-pushgateway.enabled=false

oc rollout status deployment/prometheus-lab-server -n ${OC_NS} --timeout=5m
```

Give the Prometheus service account read access to the engine namespaces, so Kubernetes service discovery finds the engine pods:

```bash
for ns in ${ENGINE_NS_A} ${ENGINE_NS_B}; do
  oc create rolebinding prometheus-read-${ns} -n ${ns} \
    --clusterrole=view \
    --serviceaccount=${OC_NS}:prometheus-lab-server
done
```

Open the Prometheus API for the sanity tests and the later phases:

```bash
oc port-forward svc/prometheus-lab-server -n ${OC_NS} 9090:80 --address 127.0.0.1 &
export PROM=http://127.0.0.1:9090
```

The release `prometheus-lab` produces the service `prometheus-lab-server`.

## 1.7 Phase 1 sanity test

The Phase 1 test proves four things: the engines run, each engine answers a completion, the engine counters increase, and Prometheus scrapes the engines.

**Test 1: all engine and Prometheus pods are running**

```bash
oc get pods -n ${ENGINE_NS_A} -l inference-lab=engine
oc get pods -n ${ENGINE_NS_B} -l inference-lab=engine
oc get pods -n ${OC_NS} -l app.kubernetes.io/name=prometheus,app.kubernetes.io/component=server
```

Pass: `vllm-gemma` shows 2 pods in `llm-a-4cost2`, `sglang-llama` shows 1 pod in `llm-a-4cost2`, `vllm-gemma` shows 1 pod in `llm-b-4cost2`, and the Prometheus server pod is `Running` and `Ready`. All engine pods are `Running` and `Ready`.

**Test 2: each engine answers with a completion**

```bash
oc port-forward svc/vllm-gemma -n ${ENGINE_NS_A} 18001:8000 --address 127.0.0.1 &
curl -s http://127.0.0.1:18001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31B","prompt":"ping","max_tokens":4}' | jq -r '.usage.completion_tokens'
```

Pass: prints `4`. Repeat for `llm-b-4cost2`, using a port-forward to `svc/vllm-gemma` in `${ENGINE_NS_B}`. For SGLang, port-forward `svc/sglang-llama` in `${ENGINE_NS_A}` to `18002:30000` and send `"model":"llama"`.

**Test 3: the engine counters exist and are non-zero after Test 2**

```bash
curl -s http://127.0.0.1:18000/metrics | grep -E '^vllm:generation_tokens_total'
```

Pass: the counter is greater than zero.

**Test 4: Prometheus scrapes the engines**

```bash
curl -s "${PROM}/api/v1/query" --data-urlencode 'query=up{job="inference-engines"}' | jq -r '.data.result[].value[1]' | sort | uniq -c
```

Pass: every result is `1`, and there are at least four series (two vLLM pods in `llm-a-4cost2`, one vLLM pod in `llm-b-4cost2`, one SGLang pod). Scrapes can take up to one scrape interval (30 seconds) to appear.

**Phase 1 is complete when Tests 1 to 4 pass.**

---

# Phase 2: SaaS stub, LiteLLM gateway

Deploys the SaaS stub, LiteLLM with PostgreSQL, the model list and identity settings from plan Section 4.3 (Group A), the teams and virtual keys, and the LiteLLM scrape job in Prometheus.

## 2.1 Create the LiteLLM namespace

```bash
oc new-project ${LITELLM_NS} || oc project ${LITELLM_NS}
```

## 2.2 Deploy the SaaS stub in `litellm-4cost2`

The stub has no GPU. It is needed for G-15 and O-13. Plan Section 4.1 only requires that it return `usage`.

The stub is a 90-line standard-library Python server, saved as `../inference-attribution-lab/saas-stub.py`. It answers `/v1/chat/completions` and `/v1/completions` with fixed usage (11 prompt tokens, 7 completion tokens, overridable with `STUB_PROMPT_TOKENS` and `STUB_COMPLETION_TOKENS`), and `/health` and `/v1/models`. It binds to the pod IP from the downward API (`POD_IP`), never to all interfaces, and logs only the method and path.

The lab runs it from a Red Hat base image with the script mounted from a ConfigMap, so there is no image to build or mirror. This replaces the earlier `ghcr.io/simanadler/openai-stub:<pinned-tag>` approach, which referenced an image that does not exist and needs a build and registry login.

Save as `../inference-attribution-lab/saas-stub.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saas-stub
  namespace: litellm-4cost2
  labels: { app: saas-stub }
spec:
  replicas: 1
  selector:
    matchLabels: { app: saas-stub }
  template:
    metadata:
      labels: { app: saas-stub }
    spec:
      containers:
        - name: stub
          image: registry.redhat.io/ubi9/python-312-minimal:latest
          command: ["python3", "/app/stub.py"]
          env:
            - name: POD_IP                      # the stub binds to this address, never to all interfaces
              valueFrom: { fieldRef: { fieldPath: status.podIP } }
            - name: PYTHONDONTWRITEBYTECODE
              value: "1"
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet: { path: /health, port: 8080 }
            periodSeconds: 5
          resources:
            requests: { cpu: 10m, memory: 32Mi }
            limits: { cpu: 200m, memory: 128Mi }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
            runAsNonRoot: true
            seccompProfile: { type: RuntimeDefault }
          volumeMounts:
            - name: src
              mountPath: /app
              readOnly: true
      volumes:
        - name: src
          configMap: { name: saas-stub-src }
---
apiVersion: v1
kind: Service
metadata:
  name: saas-stub
  namespace: litellm-4cost2
spec:
  selector: { app: saas-stub }
  ports:
    - port: 8080
      targetPort: 8080
```

```bash
oc create configmap saas-stub-src -n ${LITELLM_NS} \
  --from-file=stub.py=../inference-attribution-lab/saas-stub.py --dry-run=client -o yaml | oc apply -f -
oc apply -f ../inference-attribution-lab/saas-stub.yaml
oc rollout status deployment/saas-stub -n ${LITELLM_NS} --timeout=3m
```

> **Base image.** `ubi9/python-311-minimal` is not published in the Red Hat catalog, so the stub uses `ubi9/python-312-minimal`. `registry.redhat.io` needs credentials, and the cluster's global pull secret already holds them, so no pull secret is needed in the namespace. Record the image digest in the run report: `oc get pod -l app=saas-stub -n ${LITELLM_NS} -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'`.
>
> **Check it answers.** The stub binds to the pod IP, so `127.0.0.1` inside the pod does not reach it; use `POD_IP` instead: `oc exec deploy/saas-stub -n ${LITELLM_NS} -- python3 -c "import urllib.request,json,os;r=urllib.request.Request('http://'+os.environ['POD_IP']+':8080/v1/chat/completions',data=b'{\"model\":\"saas-stub\"}',headers={'Content-Type':'application/json'});print(json.load(urllib.request.urlopen(r))['usage'])"`. Expect `{'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}`.

## 2.3 Generate secrets

Generate every secret in the cluster. Do not write them to a file in the repo:

```bash
LITELLM_MASTER_KEY="sk-$(openssl rand -hex 24)"
POSTGRES_PASSWORD="$(openssl rand -hex 24)"
LITELLM_METRICS_SCRAPE_TOKEN="${LITELLM_MASTER_KEY}"   # Prometheus scrapes /metrics/ with the master key
HMAC_SECRET="$(openssl rand -hex 32)"                  # reserved for C9 / keyed user hash

oc create secret generic litellm-secrets -n ${LITELLM_NS} \
  --from-literal=LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}" \
  --from-literal=LITELLM_SALT_KEY="$(openssl rand -hex 32)" \
  --from-literal=UI_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=DATABASE_URL="postgresql://litellm:${POSTGRES_PASSWORD}@litellm-postgres:5432/litellm" \
  --from-literal=HMAC_SECRET="${HMAC_SECRET}"

oc create secret generic litellm-postgres -n ${LITELLM_NS} \
  --from-literal=POSTGRESQL_USER=litellm \
  --from-literal=POSTGRESQL_PASSWORD="${POSTGRES_PASSWORD}" \
  --from-literal=POSTGRESQL_DATABASE=litellm

# Prometheus scrape credential, stored in the opencost2 namespace where Prometheus runs
oc create secret generic litellm-scrape-credential -n ${OC_NS} \
  --from-literal=token="${LITELLM_METRICS_SCRAPE_TOKEN}"
```

Verify the three secrets exist and hold the expected keys:

```bash
oc get secret litellm-secrets litellm-postgres -n ${LITELLM_NS}
oc get secret litellm-scrape-credential -n ${OC_NS}
oc get secret litellm-secrets -n ${LITELLM_NS} -o json | jq -r '.data|keys[]'
# expect: DATABASE_URL, HMAC_SECRET, LITELLM_MASTER_KEY, LITELLM_SALT_KEY, UI_PASSWORD
```

> **Re-creating the variables in a new shell.** The shell variables above exist only in the shell that ran the block. If you ran it in a different shell, or opened a new one, read the master key back from the cluster before Step 2.7 and Phase 4. The other generated values are not needed again:
>
> ```bash
> export LITELLM_MASTER_KEY=$(oc get secret litellm-secrets -n ${LITELLM_NS} -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d)
> ```
>
> **Re-running this step.** `oc create secret` fails if a secret already exists. To regenerate, delete the three secrets first (`oc delete secret litellm-secrets litellm-postgres -n ${LITELLM_NS}` and `oc delete secret litellm-scrape-credential -n ${OC_NS}`). The PostgreSQL password is stored in the PostgreSQL volume on first start, so if Step 2.4 has already run, also delete the `litellm-postgres-data` PVC before regenerating.

> Use a dedicated scrape token instead of the master key if your LiteLLM version supports it. Create it through the admin API after Step 2.7 and replace the value in `litellm-scrape-credential`.

## 2.4 Deploy PostgreSQL

Use the Red Hat-supported image, which complies with the container policy. Save as `../inference-attribution-lab/postgres.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-postgres
  namespace: litellm-4cost2
  labels: { app: litellm-postgres }
spec:
  replicas: 1
  selector:
    matchLabels: { app: litellm-postgres }
  template:
    metadata:
      labels: { app: litellm-postgres }
    spec:
      containers:
        - name: postgres
          image: registry.redhat.io/rhel9/postgresql-16:latest
          ports:
            - containerPort: 5432
          envFrom:
            - secretRef: { name: litellm-postgres }
          volumeMounts:
            - name: data
              mountPath: /var/lib/pgsql/data
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
            seccompProfile: { type: RuntimeDefault }
      volumes:
        - name: data
          persistentVolumeClaim: { claimName: litellm-postgres-data }
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: litellm-postgres-data
  namespace: litellm-4cost2
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests: { storage: 10Gi }
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-postgres
  namespace: litellm-4cost2
spec:
  selector: { app: litellm-postgres }
  ports:
    - port: 5432
      targetPort: 5432
```

```bash
oc apply -f ../inference-attribution-lab/postgres.yaml
oc rollout status deployment/litellm-postgres -n ${LITELLM_NS} --timeout=5m
```

Check that the PVC is bound and that the `litellm` user can log in with the generated password. `pg_isready` and `psql` run inside the pod over `127.0.0.1`, so nothing is exposed:

```bash
oc get pvc litellm-postgres-data -n ${LITELLM_NS}          # expect STATUS Bound
oc exec deploy/litellm-postgres -n ${LITELLM_NS} -- pg_isready -h 127.0.0.1
oc exec deploy/litellm-postgres -n ${LITELLM_NS} -- psql -h 127.0.0.1 -U litellm -d litellm -c 'select 1'
```

Expect `accepting connections` and a one-row result. The pod runs under the `restricted-v2` SCC with an assigned UID, which the Red Hat PostgreSQL image supports, so no SCC change is needed.

## 2.5 Write the LiteLLM configuration

This implements Group A of plan Section 4.3 (A1 to A10). Save as `../inference-attribution-lab/litellm-config.yaml`:

```yaml
model_list:
  # Plain alias, served by vLLM in llm-a-4cost2
  - model_name: gemma-4-31B
    litellm_params:
      model: hosted_vllm/gemma-4-31B                 # A7: must equal engine --served-model-name
      api_base: http://vllm-gemma.llm-a-4cost2.svc.cluster.local:8000/v1
      api_key: "none"

  # Same served model in the second namespace (O-09, O-08)
  - model_name: gemma-4-31B-b
    litellm_params:
      model: hosted_vllm/gemma-4-31B
      api_base: http://vllm-gemma.llm-b-4cost2.svc.cluster.local:8000/v1
      api_key: "none"

  # Rewrite alias: client asks for "llama-chat", router serves "llama" (G-04)
  - model_name: llama-chat
    litellm_params:
      model: openai/llama
      api_base: http://sglang-llama.llm-a-4cost2.svc.cluster.local:30000/v1
      api_key: "none"

  # SaaS stub upstream, no GPU (G-15, O-13)
  - model_name: saas-stub
    litellm_params:
      model: openai/saas-stub
      api_base: http://saas-stub.litellm-4cost2.svc.cluster.local:8080/v1
      api_key: "none"

litellm_settings:
  callbacks: ["otel", "prometheus"]                                # A1
  enable_end_user_cost_tracking_prometheus_only: true              # A2
  prometheus_end_user_metrics_max_series_per_metric: 10000         # A3, above the lab user count
  prometheus_end_user_metrics_ttl_seconds: 86400                   # A3
  prometheus_exclude_labels: ["user_email", "hashed_api_key", "model_id", "api_provider"]  # A4
  cache: false                                                     # A9

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  require_auth_for_metrics_endpoint: true                          # A8
```

> **Verify before use (P-09a).** Setting names are taken from plan Section 4.3 and must be checked against the pinned image. An unknown setting fails at startup, which is the test you want.

## 2.6 Deploy LiteLLM with two replicas

Two replicas are needed for O-11 and G-16 (A10). Save as `../inference-attribution-lab/litellm.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: litellm-4cost2
  labels: { app: litellm }
spec:
  replicas: 2
  selector:
    matchLabels: { app: litellm }
  template:
    metadata:
      labels: { app: litellm }
    spec:
      containers:
        - name: litellm
          # Pinned to v1.108.4, the version curvebender runs and the plan's measured baseline.
          # Mirror of ghcr.io/berriai/litellm-database@sha256:cb251f2184386110fc61e4ed98a28117ab079dbf1546f212c1738805a34c6768 (see the mirror step below).
          image: ghcr.io/simanadler/litellm-database:v1.108.4
          args: ["--config", "/app/config/litellm-config.yaml", "--port", "4000"]
          ports:
            - containerPort: 4000
          envFrom:
            - secretRef: { name: litellm-secrets }
          env:
            - name: LITELLM_LOG
              value: "INFO"
          readinessProbe:
            httpGet: { path: /health/readiness, port: 4000 }
            initialDelaySeconds: 20
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health/liveliness, port: 4000 }
            initialDelaySeconds: 60
            periodSeconds: 20
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits: { memory: "2Gi" }
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: { drop: ["ALL"] }
            runAsNonRoot: true
            seccompProfile: { type: RuntimeDefault }
          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
      volumes:
        - name: config
          configMap: { name: litellm-config }
---
apiVersion: v1
kind: Service
metadata:
  name: litellm
  namespace: litellm-4cost2
spec:
  selector: { app: litellm }
  ports:
    - port: 4000
      targetPort: 4000
```

Mirror the pinned image into the lab's registry first. `imagetools create` copies the multi-arch index registry to registry without pulling the layers locally, and the digest reference guarantees the mirror matches what was verified:

```bash
docker buildx imagetools create \
  --tag ghcr.io/${GHCR_USERNAME}/litellm-database:v1.108.4 \
  ghcr.io/berriai/litellm-database@sha256:cb251f2184386110fc61e4ed98a28117ab079dbf1546f212c1738805a34c6768
docker buildx imagetools inspect ghcr.io/${GHCR_USERNAME}/litellm-database:v1.108.4 | grep -m1 Digest   # expect the same sha256:cb251f21...
```

A new ghcr.io package is private by default, so `litellm-database` needs a pull secret in `${LITELLM_NS}` before the rollout. Step 3.2 creates one for `${OC_NS}` only, and it comes later, so create this one now with a token from the environment (never a file):

```bash
oc create secret docker-registry ghcr-secret -n ${LITELLM_NS} \
  --docker-server=ghcr.io \
  --docker-username=${GHCR_USERNAME} \
  --docker-password="${GITHUB_TOKEN}"
oc secrets link default ghcr-secret --for=pull -n ${LITELLM_NS}
```

Alternatively, make the `litellm-database` package public in the GitHub package settings, which needs no secret because the image holds only the upstream LiteLLM code.

Create the ConfigMap and deploy:

```bash
oc create configmap litellm-config -n ${LITELLM_NS} \
  --from-file=litellm-config.yaml=../inference-attribution-lab/litellm-config.yaml

oc apply -f ../inference-attribution-lab/litellm.yaml
oc rollout status deployment/litellm -n ${LITELLM_NS} --timeout=10m
```

> **Why v1.108.4 and not the latest stable.** This lab deliberately pins LiteLLM to `v1.108.4` because that is the version curvebender runs, so the results transfer to the deployment that matters. It is a documented exception to the "latest stable" rule in the test plan: newer patch releases exist (`v1.108.11` and `v1.110.0` at the time of writing). On 2026-11-19, one of the 19 published GitHub security advisories for `BerriAI/litellm` includes `1.108.4` in its vulnerable range: **GHSA-2rwg-2wh4-7c4f** (high, CVSS 7.7, published 2026-11-12, fixed in 1.108.6). It lets an authenticated caller read proxy environment variables, including `LITELLM_MASTER_KEY`, by sending Azure Entra credential parameters in the request body. It applies only when an `azure/` or `azure_ai/` model is configured and no `AZURE_API_KEY` or `AZURE_OPENAI_API_KEY` is set. **Accepted risk (decision 2026-11-19): stay on `1.108.4`.** Mitigation: the lab's model list has no `azure/` or `azure_ai/` entries, and LiteLLM is reachable only inside the cluster and through a `127.0.0.1` port-forward. Do not add an Azure model to `litellm-config.yaml` while pinned to this version, and upgrade to 1.108.6 or later if that changes. The other 18 advisories do not cover `1.108.4`. Re-check at run time with `curl -s "https://api.github.com/repos/BerriAI/litellm/security-advisories?per_page=100"`, and also scan the image with the project's vulnerability scanner, since advisories do not cover the base image or Python dependencies. Record the outcome with the digest in the run report. An open critical or high CVE is a reason to stop and raise it, not to continue silently. Step P-09 of the test plan re-verifies every attribution setting name against this version, so a settings mismatch shows up there, not later.

## 2.7 Create teams and virtual keys

The key matrix (plan Section 4.1) needs one key per workload, with a team alias (tenant) and a key alias (workload). Port-forward the admin API:

```bash
oc port-forward svc/litellm 4000:4000 -n ${LITELLM_NS} --address 127.0.0.1 &
LITELLM_URL=http://127.0.0.1:4000
```

Create the teams. The names below come from the plan:

```bash
create_team() {  # $1 = team_alias
  curl -sS -X POST "${LITELLM_URL}/team/new" \
    -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"team_alias\": \"$1\"}"
}

create_key() {   # $1 = key_alias (workload), $2 = team_id
  curl -sS -X POST "${LITELLM_URL}/key/generate" \
    -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"key_alias\": \"$1\", \"team_id\": \"$2\", \"models\": []}"
}

create_team "acme-corp/engineering"
create_team "acme-corp"
create_team "globex"
create_team "limited"
```

Read the returned `team_id` values with `jq -r .team_id`, then call `create_key` for each workload:

| Key alias (workload) | Team alias (tenant) |
|---|---|
| `rag` | `acme-corp/engineering` |
| `batch` | `acme-corp/engineering` |
| `engineering/rag` | `acme-corp` |
| `chat` | `globex` |
| `agent` | `globex` |

The `noteam-key` (no team, no key alias) and `limited` (rate-limit and budget) keys are created separately:

```bash
curl -sS -X POST "${LITELLM_URL}/key/generate" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" -H "Content-Type: application/json" \
  -d '{}'                                                    # noteam-key: no team, no alias (G-03)
```

For `limited`, set `rpm_limit` and `max_budget` on the `/key/generate` call, for example `{"key_alias":"limited","team_id":"<id>","rpm_limit":1,"max_budget":0.0001}`.

Store the generated key values in the cluster, never in files. Read them with `read -s` so they stay out of shell history and files:

```bash
oc create secret generic litellm-virtual-keys -n ${OC_NS} \
  --from-literal=rag="<rag-key>" --from-literal=batch="<batch-key>" \
  --from-literal=engineering-rag="<engineering/rag-key>" \
  --from-literal=chat="<chat-key>" --from-literal=agent="<agent-key>" \
  --from-literal=noteam-key="<noteam-key>" --from-literal=limited="<limited-key>"
```

Replace the `<…-key>` values with the generated keys.

## 2.8 Add the LiteLLM scrape job to Prometheus

Phase 2 adds the LiteLLM job. Save the complete values file as `../inference-attribution-lab/prometheus-values-phase2.yaml`. It keeps the engine job and the OpenShift block from Phase 1 (Step 1.6.1), adds the LiteLLM job, mounts the scrape token, and raises the server volume to 20Gi. It repeats every block from Step 1.6.1 (engine relabels, `scrapeConfigs`, retention, the kube-state-metrics allowlist, the SCC settings). `--reuse-values` does not make a partial file safe: without the SCC blocks the upgrade would reapply the chart's hardcoded 65534 IDs and the server pod would be rejected.

```yaml
extraScrapeConfigs: |
  # Engine counters (vllm:* and sglang:*). honor_labels keeps model_name intact (P-04).
  # Scrapes each pod's declared container port (vLLM 8000, SGLang 30000); see Step 1.6.1.
  - job_name: inference-engines
    honor_labels: true
    scrape_interval: 30s
    metrics_path: /metrics
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [llm-a-4cost2, llm-b-4cost2]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_inference_lab]
        regex: engine
        action: keep
      # OpenCost joins engine tokens to model cost on model_name:namespace, so the engine
      # series must carry the pod's namespace. Without it they are filed under
      # namespace="unknown" and never match the cost entry for the real namespace.
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

  # LiteLLM counters, plan A8: /metrics/ with trailing slash, authenticated
  - job_name: litellm
    honor_labels: true
    scrape_interval: 30s
    metrics_path: /metrics/
    scheme: http
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus-secrets/litellm/token
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: [litellm-4cost2]
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        regex: litellm
        action: keep
      # The litellm Service port is unnamed, so match on the target port in __address__.
      - source_labels: [__address__]
        regex: ".*:4000"
        action: keep

# Replace the chart's default cluster-wide scrape jobs. The lab only needs the engine and
# LiteLLM jobs above (extraScrapeConfigs), plus the cadvisor container metrics and the
# kube-state-metrics that OpenCost's allocation queries read, limited to the lab namespaces.
# The default cluster-wide scrape filled the 20Gi volume in about 3 days and stopped
# Prometheus (write to WAL: disk quota exceeded).
scrapeConfigs:
  kubernetes-api-servers:
    enabled: false
  kubernetes-nodes:
    enabled: false
  kubernetes-service-endpoints:
    enabled: false
  kubernetes-service-endpoints-slow:
    enabled: false
  prometheus-pushgateway:
    enabled: false
  kubernetes-services:
    enabled: false
  kubernetes-pods:
    enabled: false
  kubernetes-pods-slow:
    enabled: false
  # Keep Prometheus scraping itself (the chart's own default job, unchanged).
  prometheus:
    enabled: true
  # cadvisor is scraped per node, so keep only series from the lab namespaces at ingest.
  kubernetes-nodes-cadvisor:
    enabled: true
    scheme: https
    metrics_path: /metrics/cadvisor
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      insecure_skip_verify: true
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - source_labels: [__metrics_path__]
        target_label: metrics_path
    metric_relabel_configs:
      - source_labels: [namespace]
        regex: (llm-a-4cost2|llm-b-4cost2|litellm-4cost2)
        action: keep
  # kube-state-metrics from this release, for node, pod and PVC info.
  kube-state-metrics:
    enabled: true
    honor_labels: true
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: [opencost2]
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        regex: prometheus-lab-kube-state-metrics
        action: keep
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        regex: http
        action: keep

# OpenShift restricted-v2 SCC: let the platform assign UID/GID/fsGroup from the
# namespace range instead of the chart's hardcoded 65534.
server:
  securityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null
    runAsNonRoot: true
  # The 8Gi default filled up (WAL), which crash-looped the server. Expandable storage class.
  persistentVolume:
    size: 20Gi
  # Bound the TSDB so it cannot fill the volume again.
  retention: 3d
  retentionSize: 12GB
  # Mounts the LiteLLM scrape token for credentials_file above.
  extraSecretMounts:
    - name: litellm-scrape
      secretName: litellm-scrape-credential
      mountPath: /etc/prometheus-secrets/litellm
      readOnly: true
kube-state-metrics:
  securityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null
    runAsNonRoot: true
  # kube-state-metrics v2 exports no pod labels unless allowlisted. OpenCost reads pod labels
  # from kube_pod_labels to group cost by model; without this every pod lands in
  # __unallocated__ and the model cost is $0. The label values must equal the engine's
  # model_name (--served-model-name).
  metricLabelsAllowlist:
    - pods=[llm-d.ai/model,llm-d.ai/inference-shared]
# node-exporter needs hostNetwork/hostPID/hostPath, which no SCC usable here allows.
prometheus-node-exporter:
  enabled: false
```

`extraSecretMounts` is a key under `server` in this chart (confirm with `helm show values prometheus-community/prometheus | grep -n extraSecretMounts`), not a top-level key.

If the Prometheus volume from Phase 1 is already near full, expand it before the upgrade. A new server pod on a full volume crash-loops and needs WAL repair on restart:

```bash
oc get sc -o custom-columns=NAME:.metadata.name,EXPAND:.allowVolumeExpansion   # EXPAND must be true for the PVC's class
oc patch pvc prometheus-lab-server -n ${OC_NS} -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

Give the Prometheus service account read access to the LiteLLM namespace:

```bash
oc create rolebinding prometheus-read-${LITELLM_NS} -n ${LITELLM_NS} \
  --clusterrole=view \
  --serviceaccount=${OC_NS}:prometheus-lab-server
```

Upgrade the release in place, reusing the same release name:

```bash
helm upgrade prometheus-lab prometheus-community/prometheus \
  -n ${OC_NS} \
  -f ../inference-attribution-lab/prometheus-values-phase2.yaml \
  --reuse-values \
  --set alertmanager.enabled=false \
  --set kube-state-metrics.enabled=true \
  --set prometheus-pushgateway.enabled=false
oc rollout status deployment/prometheus-lab-server -n ${OC_NS} --timeout=5m
```

If the rollout stalls, `oc logs deploy/prometheus-lab-server -c prometheus-server --previous` shows the cause. After a volume-full recovery Prometheus compacts a backlog and its API can be unresponsive for several minutes.

Check that both LiteLLM replicas are scraped (`up` is 1 for each; the `inference-engines` series are checked in Step 1.7):

```bash
curl -s "${PROM}/api/v1/query" --data-urlencode 'query=up{job="litellm"}' | jq -r '.data.result[] | [.metric.instance,.value[1]] | @tsv'
```

Expect two lines, one per LiteLLM pod, each ending in `1`. The `litellm_*` token counters appear after the first request through the gateway (Step 2.9).

## 2.9 Phase 2 sanity test

**Test 1: SaaS stub, LiteLLM and PostgreSQL pods are ready**

```bash
oc get pods -n ${LITELLM_NS} -l app=saas-stub
oc get pods -n ${LITELLM_NS} -l app=litellm
oc get pods -n ${LITELLM_NS} -l app=litellm-postgres
```

Pass: one stub pod, two LiteLLM pods and one Postgres pod, all `Running` and `Ready`.

**Test 2: the health endpoint answers**

```bash
curl -s -o /dev/null -w '%{http_code}\n' "${LITELLM_URL}/health/readiness"
```

Pass: `200`.

**Test 3: an authenticated chat request reaches the engine and returns usage**

```bash
export LITELLM_KEY_RAG="<rag-key>"       # set from the litellm-virtual-keys secret, not inline
curl -s "${LITELLM_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${LITELLM_KEY_RAG}" -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31B","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"user":"alice"}' | jq '.usage'
```

Pass: `usage` is present. The request is counted by LiteLLM and by the engine.

**Test 4: an unauthenticated request is rejected**

```bash
curl -s -o /dev/null -w '%{http_code}\n' "${LITELLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31B","messages":[{"role":"user","content":"ping"}]}'
```

Pass: `401`.

**Test 5: the metrics endpoint needs authentication and exposes the token counters**

```bash
curl -s -o /dev/null -w '%{http_code}\n' "${LITELLM_URL}/metrics/"                       # expect 401
curl -s -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" "${LITELLM_URL}/metrics/" \
  | grep -E '^litellm_(input|output)_tokens_metric_total' | head -5
```

Pass: the first command prints `401`. The second lists the counters after Test 3 has run. Record the label set in the run report; this is the P-02 census.

**Test 6: Prometheus scrapes LiteLLM**

```bash
curl -s "${PROM}/api/v1/query" --data-urlencode 'query=up{job="litellm"}' | jq -r '.data.result[].value[1]'
```

Pass: `1` for each LiteLLM pod. Allow one scrape interval (30 seconds) after the Prometheus upgrade.

**Test 7: no configuration error at startup (P-09a)**

```bash
oc logs deployment/litellm -n ${LITELLM_NS} --tail=300 | grep -i -E 'unknown|invalid' || echo "no unknown or invalid settings"
```

Pass: prints `no unknown or invalid settings`.

**Test 8: the SaaS stub answers with usage**

```bash
oc port-forward svc/saas-stub -n ${LITELLM_NS} 18080:8080 --address 127.0.0.1 &
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"saas-stub","messages":[{"role":"user","content":"ping"}]}' | jq '.usage'
```

Pass: `usage` is present with `prompt_tokens` and `completion_tokens`.

**Phase 2 is complete when Tests 1 to 8 pass.**

---

# Phase 3: OpenCost

Installs the OpenCost image built from the current branch. Before you build, confirm the required PRs are in the branch. If they are not, you can incorporate them as described in [Appendix A](#appendix-a-incorporating-the-opencost-prs-into-the-branch). Appendix A is optional for Phases 3 and 4. Without #4161 the SGLang queries are missing, and without #4163 model cost is not keyed by namespace. SGLang and per-namespace-cost assertions, and Phase 0 check P-01, are then expected to fail. The vLLM attribution path in this guide works without them.

The engines, Prometheus and LiteLLM are already running, so the attribution health check reads real series. Until the Step 4.2 census confirms the label names, the health check may warn about them. That is expected.

## 3.1 Build and push the OpenCost image

The repository's `Dockerfile.openshift` builds the `costmodel` binary from the working tree for `linux/amd64`. Confirm the branch check in Step 0.3 first.

```bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.openshift \
  -t ${OPENCOST_IMAGE} \
  --build-arg version=inference-v2 \
  --build-arg commit=$(git rev-parse --short HEAD) \
  --push \
  .
```

Record the digest. The run report must include it (plan Section 12):

```bash
docker buildx imagetools inspect ${OPENCOST_IMAGE} | grep -i digest | head -1 | tee ../inference-attribution-lab/opencost-image-digest.txt
```

> **Go version.** `go.mod` requires Go 1.27.3 or later, so the builder stage must be `golang:1.27-alpine` or newer. With `golang:1.26-alpine` the build fails at `go mod download` with `go.mod requires go >= 1.27.3`. Update line 5 of `Dockerfile.openshift` if it still says 1.26.

> **Security deviation to record.** `Dockerfile.openshift` uses `golang:1.27-alpine` and `alpine:latest`. The project security rules require minimal base images from `registry.redhat.io` (for example `ubi9/go-toolset` and `ubi9/ubi-minimal`). For a lab run, record this as a documented deviation in the run report. For a production image, change both `FROM` lines to the Red Hat images before you build.

> **Build time.** On a workstation that is not `linux/amd64` (for example Apple silicon under Rancher Desktop), the build runs under emulation. Expect the `go build` step to take 10 to 20 minutes, and run the build in the background and watch its log. A transient `apk add` I/O error is a network or VM disk problem; re-run the build.

Confirm the image runs. The binary has no `--version` flag, so use `--help`:

```bash
docker run --rm --platform linux/amd64 ${OPENCOST_IMAGE} --help | head -3
```

Expect `Cost-Model metric exporter and data aggregator.`

## 3.2 Create the pull secret

The image is private to the `simanadler` ghcr.io account. Create a pull secret in the lab namespace, using a token from the environment. Do not copy the `ghcr-secret` from the `opencost` namespace; the lab keeps its own credentials:

```bash
oc create secret docker-registry ghcr-secret -n ${OC_NS} \
  --docker-server=ghcr.io \
  --docker-username=${GHCR_USERNAME} \
  --docker-password="${GITHUB_TOKEN}"
oc secrets link default ghcr-secret --for=pull -n ${OC_NS}
```

## 3.3 Create the attribution configuration and admin token

### 3.3.1 Attribution configuration

Save as `../inference-attribution-lab/attribution-config.yaml`. This is the baseline V0 mapping from plan Section 4.2. The `workloadLabel` and `userLabel` values are hypotheses until check P-08 confirms them in Step 4.2. Update them there if the names differ:

```yaml
producers:
  - name: litellm-gateway
    inputTokensSeries: litellm_input_tokens_metric_total
    outputTokensSeries: litellm_output_tokens_metric_total
    tenantLabel: team_alias
    workloadLabel: api_key_alias      # hypothesis; confirm with P-08
    userLabel: end_user               # hypothesis; confirm with P-08
    modelLabel: model
    namespaceLabel: deployment_namespace
    identitySentinel: "None"          # confirm with G-03
    userSentinel: "None"
    scope:
      servingModels: [gemma-4-31B]
```

```bash
oc create configmap opencost-attribution -n ${OC_NS} \
  --from-file=attribution.yaml=../inference-attribution-lab/attribution-config.yaml
```

### 3.3.2 Admin token

```bash
oc create secret generic opencost-admin -n ${OC_NS} \
  --from-literal=ADMIN_TOKEN="$(openssl rand -hex 32)"
```

## 3.4 Install OpenCost

### 3.4.1 Helm values

Save as `../inference-attribution-lab/opencost-values.yaml`. The Prometheus endpoint is the `prometheus-lab` release from Step 1.6:

```yaml
imagePullSecrets:                 # top level in chart 2.5.12, not under exporter
  - name: ghcr-secret

extraVolumes:                     # top level in chart 2.5.12, not under exporter
  - name: attribution
    configMap:
      name: opencost-attribution

opencost:
  prometheus:
    internal:                     # sets PROMETHEUS_SERVER_ENDPOINT; do not set it in extraEnv
      enabled: true
      serviceName: prometheus-lab-server
      namespaceName: opencost2
      port: 80
  exporter:
    image:
      registry: ghcr.io
      repository: simanadler/opencost-inference
      tag: inference-v2
      pullPolicy: Always
    defaultClusterId: opencost2-lab
    extraEnv:                     # a map (NAME: value), not a list
      INFERENCE_COST_ENABLED: "true"
      INFERENCE_COLLECTION_INTERVAL: "60s"
      INFERENCE_ATTRIBUTION_ENABLED: "true"
      INFERENCE_ATTRIBUTION_CONFIG_PATH: /etc/opencost-attribution/attribution.yaml
      INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED: "true"
    adminToken:
      enabled: true
      existingSecret: opencost-admin
      secretKey: ADMIN_TOKEN
    extraVolumeMounts:
      - name: attribution
        mountPath: /etc/opencost-attribution
        readOnly: true
  ui:
    # The upstream UI image cannot write /var/www under the OpenShift-assigned UID
    # (CrashLoopBackOff). The test uses only the API on 9003.
    enabled: false
```

> **Chart keys verified against chart 2.5.12** (the version of the live `opencost` release) with `helm show values` and `helm template`. The rendered Deployment carries the image, pull secret, `PROMETHEUS_SERVER_ENDPOINT`, the `INFERENCE_*` variables, `ADMIN_TOKEN` from `opencost-admin`, and the attribution ConfigMap mount. If you use another chart version, re-check the key names.

Note the variable names. `INFERENCE_COLLECTION_INTERVAL` is the name defined in `pkg/env/costmodel.go`, and it takes a Go duration such as `60s`. A bare `60` fails to parse and silently falls back to the 2 minute default. The repository's `DEPLOY.sh` and `BUILD_PUSH_DEPLOY.sh` set `INFERENCE_COST_COLLECTION_INTERVAL`, which the code does not read.

### 3.4.2 Install

```bash
helm repo add opencost https://opencost.github.io/opencost-helm-chart
helm repo update

helm list -n ${OC_NS}          # expect no opencost2 release
helm install opencost2 opencost/opencost \
  --version 2.5.12 \
  -n ${OC_NS} \
  -f ../inference-attribution-lab/opencost-values.yaml
```

The release is named `opencost2`, not `opencost`, so it does not share a name with the existing release in the `opencost` namespace. The Deployment and Service take their names from the release.

### 3.4.3 Fix the security context for OpenShift

`Dockerfile.openshift` sets `USER 1001`. The OpenShift `restricted-v2` SCC assigns UIDs from the project range, so the pod can be rejected. Check first:

```bash
oc get pods -n ${OC_NS} -l app.kubernetes.io/name=opencost
oc get events -n ${OC_NS} --sort-by=.lastTimestamp | tail -20
```

If the events show `unable to validate against any security context constraint`, remove the fixed UID from the pod spec and let the SCC choose:

```bash
oc patch deployment opencost2 -n ${OC_NS} --type=json \
  -p '[{"op":"remove","path":"/spec/template/spec/containers/0/securityContext/runAsUser"}]' 2>/dev/null || true
```

If the SCC still rejects the pod, the image must be rebuilt without a fixed UID. Do not grant `anyuid` as the default. Record it in the run report if it is the only option.

In the verified run, this step was not needed. The pod was admitted under `restricted-v2` with an SCC-assigned UID (`1002260000`), because the chart does not set a fixed `runAsUser`.

### 3.4.4 Wait for the rollout and check the logs

```bash
oc rollout status deployment/opencost2 -n ${OC_NS} --timeout=5m
oc logs deployment/opencost2 -n ${OC_NS} --tail=200 | grep -iE 'inference|attribution|error|warn'
```

Look for no `exec format error` (architecture) and the attribution health check starting. Expect no ERROR from the health check. Missing attribution series produce warnings until the Step 4.2 traffic creates them.

## 3.5 Phase 3 sanity test

The Phase 3 test proves five things: the pod runs, OpenCost answers its API, OpenCost reaches Prometheus, the attribution configuration loaded, and the existing `opencost` install is untouched. Run these after the rollout completes.

**Test 1: pods are running and ready**

```bash
oc get pods -n ${OC_NS} -l app.kubernetes.io/name=opencost
```

Pass: the OpenCost pod shows `Running` with all containers ready.

**Test 2: OpenCost answers the model-level API**

```bash
oc port-forward svc/opencost2 -n ${OC_NS} 9003:9003 --address 127.0.0.1 &
export API=http://127.0.0.1:9003
curl -s "${API}/allocation/compute?window=1h&aggregate=namespace" | jq '.code'
```

Pass: `.code` is `200`.

**Test 3: OpenCost reaches Prometheus**

```bash
oc logs deployment/opencost2 -n ${OC_NS} --tail=300 | grep -i 'prometheus' | grep -i -E 'error|fail|unreachable' || echo "no Prometheus connection errors"
oc set env deployment/opencost2 -n ${OC_NS} --list | grep PROMETHEUS_SERVER_ENDPOINT
```

Pass: the first command prints `no Prometheus connection errors`. The second prints `PROMETHEUS_SERVER_ENDPOINT=http://prometheus-lab-server.opencost2.svc.cluster.local:80`.

**Test 4: the attribution configuration is loaded and there is no health-check ERROR**

```bash
oc logs deployment/opencost2 -n ${OC_NS} --tail=500 | grep -i 'attribution' | grep -i 'error' || echo "no attribution ERROR"
oc set env deployment/opencost2 -n ${OC_NS} --list | grep -E 'INFERENCE_(COST|ATTRIBUTION)_ENABLED|INFERENCE_ATTRIBUTION_CONFIG_PATH'
```

Pass: the first command prints `no attribution ERROR`. The second lists all three variables with the values from Step 3.4.1.

**Test 5: the lab install does not touch the existing `opencost` install**

```bash
helm list -n opencost
oc get deploy opencost -n opencost -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```

Pass: the `opencost` and `prometheus-opencost` releases are still listed, and the image is still `ghcr.io/opencost/opencost:test-763d284`.

**Phase 3 is complete when Tests 1 to 5 pass.** Keep the port-forwards running for Phase 4, or restart them as needed.

---

# Phase 4: load generator (client side)

The load generator is `tools/inference-loadgen/loadgen.py` in this repository. It is a standard-library Python 3.11+ script with no dependencies, so there is nothing to install. It sends deterministic traffic through LiteLLM, reads the engine oracle counters before and after each case, and writes a JSON record per case. The assertion runner (plan Section 12) consumes those records and is not yet written.

## 4.1 Check the tool

```bash
python3 tools/inference-loadgen/loadgen.py --help
```

Store each virtual key in an environment variable in your shell. The tool never accepts a key as an argument:

```bash
export LITELLM_KEY_RAG="$(oc get secret litellm-virtual-keys -n ${OC_NS} -o jsonpath='{.data.rag}' | base64 -d)"
```

Repeat for the other keys (`LITELLM_KEY_BATCH`, `LITELLM_KEY_CHAT`, `LITELLM_KEY_AGENT`, `LITELLM_KEY_NOTEAM`, `LITELLM_KEY_LIMITED`).

## 4.2 Run the smoke test case

Run a short case with the `rag` key. It sends a body `user` and checks the engine oracle against the client usage (plan P-02, P-08):

```bash
python3 tools/inference-loadgen/loadgen.py case \
  --case-id smoke-rag --gateway http://127.0.0.1:4000 --engine http://127.0.0.1:18000 \
  --model gemma-4-31B --key-env LITELLM_KEY_RAG --user alice \
  --requests 2 --prompt-tokens 128 --output-tokens 16 \
  --out ../inference-attribution-lab/runs/smoke-rag.json
```

The `case` command reads the engine counters through the port-forward from Step 1.5 before and after each case. It records `engine_delta_prompt` and `engine_delta_generation` in the JSON record. These are `E_in` and `E_out` (plan Section 5.2). Do not use a client-side tokenizer for this.

Then scrape the LiteLLM counters and record the label set in the run report:

```bash
curl -s -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" http://127.0.0.1:4000/metrics/ \
  | grep -E '^litellm_(input|output)_tokens_metric_total' | head -5
```

That output is the P-02 census and the input for P-08. If the label names differ from Step 3.3.1, update `attribution-config.yaml`, re-run Step 3.3.1, and run the OpenCost upgrade with `helm upgrade opencost2 opencost/opencost -n ${OC_NS} -f ../inference-attribution-lab/opencost-values.yaml`.

## 4.3 Phase 4 sanity test

**Test 1: the tool runs and prints its help**

```bash
python3 tools/inference-loadgen/loadgen.py --help | head -3
```

Pass: prints the usage line.

**Test 2: the smoke case completes with every request succeeding**

```bash
jq '{requests_sent, requests_ok}' ../inference-attribution-lab/runs/smoke-rag.json
```

Pass: `requests_ok` equals `requests_sent` (2 and 2).

**Test 3: the engine oracle matches the client usage**

```bash
jq '{engine_delta_prompt, client_usage_prompt_total, engine_delta_generation, client_usage_completion_total}' \
  ../inference-attribution-lab/runs/smoke-rag.json
```

Pass: `engine_delta_prompt` equals `client_usage_prompt_total`, and `engine_delta_generation` equals `client_usage_completion_total`. Plan Section 5.2 requires an exact match. A mismatch means the engine served traffic that the client did not send, or the port-forward points at the wrong engine.

**Test 4: the attribution counters increase in LiteLLM**

```bash
curl -s -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" http://127.0.0.1:4000/metrics/ \
  | grep -E '^litellm_input_tokens_metric_total' | awk '{s+=$NF} END {print s}'
```

Pass: the sum is greater than zero and grows after each new `case` run.

**Test 5: the record has a window**

```bash
jq '{t0, t1}' ../inference-attribution-lab/runs/smoke-rag.json
```

Pass: `t0` is earlier than `t1`, both are ISO-8601 timestamps.

**Test 6: OpenCost cost and attribution agree with the engine**

Wait two collection cycles after the case ends, then query a window that covers it. Windows of a few minutes can split a case across two collection cycles and show 0 tokens:

```bash
curl -s "${API}/inferenceCost/total?window=12m&aggregate=model_name,namespace" \
  | jq '.data.inferenceCosts | to_entries[] | select(.key|test("gemma")) | {key, totalCost: .value.totalCost, p: .value.promptTokens, g: .value.generationTokens}'
curl -s "${API}/inferenceCost/total?window=12m&aggregate=tenant_id" \
  | jq '.data.inferenceCosts | to_entries[] | {key, p: .value.promptTokens, g: .value.generationTokens}'
```

Pass: `gemma-4-31B/llm-a-4cost2` has `totalCost` above 0 and its tokens equal `engine_delta_*` from the record. The tenant `acme-corp/engineering` has the same tokens. `oc logs deploy/opencost2 -n ${OC_NS}` shows no `over-attribution` line. A row `gemma-4-31B/unknown` with tokens and 0 cost means the engine series lack the `namespace` label (Step 1.6.1). A model cost of 0 with tokens means the pod lacks the `llm-d.ai/model` label or kube-state-metrics does not export it (Steps 1.3 and 1.6.1).

> **Port-forwards.** Run the Step 1.5 and Step 2.7 port-forwards and the OpenCost port-forward in separate terminals or as background processes. A port-forward to an engine pod ends when that pod restarts, and `oc login` sessions that use SSO expire. After a restart, re-run Step 1.5 so `--engine` points at the new pod.
>
> **Attribution accuracy.** The attribution query takes the exact counter delta (`last_over_time` at the window end minus at the window start, per series). An earlier version used `increase()` over the window. `increase()` extrapolates when the scrape interval is a large fraction of the window, so with a 30 s scrape and a 60 s window it reported 2.00 times the real tokens (544 against 272). If you see attributed tokens above the engine's, check that the image digest in Step 3.1 includes this fix.

**Phase 4 is complete when Tests 1 to 6 pass.**

---

## Full verification checklist

Confirm all four phases before the Phase 0 gate checks in the plan.

- [ ] Phase 1 sanity test (Step 1.7 Tests 1 to 4) passes
- [ ] Phase 2 sanity test (Step 2.9 Tests 1 to 8) passes
- [ ] Phase 3 sanity test (Step 3.5 Tests 1 to 5) passes
- [ ] Phase 4 sanity test (Step 4.3 Tests 1 to 6) passes
- [ ] `oc get pods -n llm-a-4cost2` shows `vllm-gemma` (2), `sglang-llama` (1)
- [ ] `oc get pods -n llm-b-4cost2` shows `vllm-gemma` (1)
- [ ] `oc get pods -n litellm-4cost2` shows two LiteLLM replicas, Postgres and the SaaS stub `Running`
- [ ] `oc get pods -n opencost2` shows OpenCost and Prometheus `Running`
- [ ] Prometheus target page (`${PROM}/targets`) lists the `inference-engines` and `litellm` jobs as `UP`
- [ ] The OpenCost image digest, LiteLLM tag and digest, vLLM and SGLang versions are in the run report
- [ ] No secret appears in any file under `../inference-attribution-lab` or in the repository (`git status` and `grep -r` checks pass)

---

## Phase 0 gate checks

Run the checks in plan Section 6 in order. Stop at the first failure. Phase 0 runs after all four phases.

| ID | Quick check |
|---|---|
| P-01 | `curl -s ${API}/allocation/compute?window=1h&aggregate=namespace` works (`${API}` from Step 3.5 Test 2), and the build contains #4161 and #4163 (Step 0.3) |
| P-02 | Label census from Step 4.2 saved to the run report |
| P-03 | `curl -s -o /dev/null -w '%{http_code}' ${LITELLM_URL}/metrics/` returns `200` with the master key and `401` without it |
| P-04 | Compare the `model` label on the LiteLLM counter with `model_name` on the engine counter for `gemma-4-31B` and `llama-chat` |
| P-05 | `curl -s ${PROM}/api/v1/query --data-urlencode 'query=llm_dimension_unmatched_keys{producer="litellm-gateway"}'` returns `0` after traffic |
| P-08 | Send one request per key with distinct body `user` values, then read which counter labels hold the team, key alias and user. Update `attribution-config.yaml` if the names differ (Step 3.3.1) |
| P-09 | Confirm each Group A setting is accepted at LiteLLM startup: Step 2.9 Test 7 |

Any check that cannot run is **BLOCKED**, not skipped. Record it in the run report.

---

## Teardown

Stop the port-forwards and remove the lab. The secrets go with the namespaces. The existing `opencost` namespace is not part of the lab and must never be deleted here. The guard below stops the teardown if `OC_NS` has been set to anything other than `opencost2`:

```bash
[ "${OC_NS}" = "opencost2" ] || { echo "refusing: OC_NS=${OC_NS}"; exit 1; }
jobs -p | xargs -r kill
oc delete project ${OC_NS} ${LITELLM_NS} ${ENGINE_NS_A} ${ENGINE_NS_B}
```

Keep `../inference-attribution-lab/*.txt` and the run report until the results are committed.

---

## Open items to resolve before the first run

| # | Item | Where it is decided |
|---|---|---|
| 1 | LiteLLM is pinned to `v1.108.4` (digest `sha256:cb251f21...`); record the security-advisory check for that version | Step 2.6 |
| 2 | Hugging Face repository names for gemma and llama | Steps 1.3 and 1.4 placeholders |
| 3 | SGLang image tag and metrics port for the installed version | Step 1.4 |
| 4 | OpenCost Helm chart value keys for the installed chart version | Step 3.4.1 |
| 5 | Whether the OpenShift SCC accepts the image's `USER 1001` | Step 3.4.3 |
| 6 | Mirror digests for `vllm-openai`, `sglang`, the LiteLLM image and the OpenCost image in `ghcr.io/simanadler`, and the `registry.redhat.io` deviation for each (the SaaS stub needs none, see item 9) | Steps 1.3, 1.4, 2.6 and 3.1 |
| 7 | `workloadLabel` and `userLabel` names (P-08) | Steps 3.3.1 and 4.2 |
| 8 | Commits for #4161 and #4163 on the branch | Step 0.3 |
| 9 | The SaaS stub runs from a Red Hat base image with its script in a ConfigMap, so nothing is built or mirrored; record the base image digest | Step 2.2 |

---

## Appendix A: Incorporating the OpenCost PRs into the branch

Run this only if Step 0.3 shows that the required commits are missing. The integration test plan (Section 4.2 and the Phase 0 checks in Section 6) requires two upstream PRs in the OpenCost build:

| PR | Purpose | Commits | Merge result on `inference-v2-implementation` (checked against `24b94ef6`) |
|---|---|---|---|
| [opencost/opencost#4161](https://github.com/opencost/opencost/pull/4161) | SGLang engine queries, cluster filter for inference queries | `74593b54`, `3ba341bd`, `1bc356b5` | Conflicts in `modules/prometheus-source/pkg/prom/inference_queries.go` |
| [opencost/opencost#4163](https://github.com/opencost/opencost/pull/4163) | Model cost keyed by model label and namespace (fixes #4162) | `84119ed8` | Applies cleanly |

PR #4162 is an issue, not a PR. #4163 is the fix for it, so there is nothing separate to incorporate.

The commits are from the upstream PR heads. Confirm the SHAs at run time, because a PR can be rebased.

### A.1 Fetch the PR heads

```bash
cd opencost
git fetch https://github.com/opencost/opencost.git pull/4161/head pull/4163/head
git log --oneline HEAD..FETCH_HEAD                    # review what will be added
```

Record the fetched SHAs, which become part of the run report:

```bash
git rev-parse 1bc356b5 84119ed8 > ../inference-attribution-lab/pr-heads.txt
```

### A.2 Integrate #4163

It applies without conflict. Cherry-pick it onto a dedicated branch so the run can be reverted cleanly:

```bash
git switch -c inference-v2-with-prs
git cherry-pick 84119ed8
```

### A.3 Integrate #4161, resolving the conflict

PR #4161 conflicts with the branch in `inference_queries.go`. Cherry-pick the three commits in order and resolve the conflict by hand:

```bash
git cherry-pick 74593b54 3ba341bd 1bc356b5
```

When the cherry-pick stops, the conflicted file is `modules/prometheus-source/pkg/prom/inference_queries.go`:

```bash
git status --short | grep '^UU'
```

Resolve each conflict block so the SGLang query functions from #4161 are kept and the branch's own query changes are kept. Then mark it resolved and continue:

```bash
git add modules/prometheus-source/pkg/prom/inference_queries.go
git cherry-pick --continue
```

Do not resolve it by taking one side wholesale. `inference_queries_test.go` auto-merges and must still compile.

### A.4 Verify the result

```bash
go build ./...
go test ./pkg/inferencecost/... ./modules/prometheus-source/pkg/prom/...
git log --oneline -6                                   # the three #4161 commits and #4163 are present
```

Then confirm P-01 in Phase 0 against the built image. Use `${OPENCOST_IMAGE}`, built from this branch in Step 3.1, and not an image from before the cherry-picks.

### A.5 Record the deviation

The branch is no longer `inference-v2-implementation` as it was before this appendix. Record the new branch name, the base commit `24b94ef6`, and the PR head SHAs in the run report. Update the build commit in Step 0.3 to match the image you build.
