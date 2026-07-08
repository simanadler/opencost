#!/usr/bin/env bash
#
# OpenCost Inference Cost — LIVE DEMO
# -----------------------------------
# Run this DURING the demo. It reads the pre-selected models from .demo-models
# (produced by inference-demo-setup.sh, run privately beforehand), shows a short
# intro, then walks each selected model through the inference REST APIs one call
# at a time:
#
#   For each model, for each API call:
#     - the curl being run
#     - the raw JSON response
#     - a human-readable rendering
#     - a plain-English "what this means" note
#     - pause for <Enter>
#
# Ends with an overall summary across the selected models.
#
# This script only ever mentions the pre-selected models. Model discovery and
# any skipped/mislabeled workloads live entirely in the setup script.
#
# Requirements: bash, curl, jq. OpenCost API reachable at $API (port-forward 9003).
#
# Usage:
#   ./inference-demo.sh                       # reads ./.demo-models, 24h window
#   WINDOW=6h ./inference-demo.sh
#   MODELSFILE=/tmp/demo-models ./inference-demo.sh
#   MODELS="gemma-4-31B-it,Llama-3.3-70B-Instruct-FP8-dynamic" ./inference-demo.sh
#
set -uo pipefail

# ---- Config ---------------------------------------------------------------
API="${API:-http://localhost:9003}"
WINDOW="${WINDOW:-24h}"
ACCUMULATE="${ACCUMULATE:-hour}"      # timeseries step
CURL_TIMEOUT="${CURL_TIMEOUT:-120}"
MODELSFILE="${MODELSFILE:-.demo-models}"

# ---- Colors / formatting --------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  CYAN=$'\033[36m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; MAGENTA=$'\033[35m'
else
  BOLD=""; DIM=""; RESET=""; CYAN=""; GREEN=""; YELLOW=""; MAGENTA=""
fi

DASH="----------------------------------------------------------------------"
hr()      { printf '%s\n' "$DASH"; }
banner()  { echo; hr; echo "${BOLD}${CYAN}$*${RESET}"; hr; }
section() { echo; echo "${BOLD}${MAGENTA}$*${RESET}"; }
label()   { echo "${DIM}$*${RESET}"; }
pause()   { echo; read -r -p "${DIM}   -- press <Enter> to continue --${RESET}" _ignore; echo; }

apiget() { curl -s --max-time "$CURL_TIMEOUT" "${API}$1"; }

# ---- Preflight ------------------------------------------------------------
command -v jq   >/dev/null 2>&1 || { echo "ERROR: jq not found";   exit 1; }
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }

# ---- Load the pre-selected models (no discovery, no bad-model mentions) ----
DEMO_MODELS=()
if [ -n "${MODELS:-}" ]; then
  OLDIFS="$IFS"; IFS=','; for m in $MODELS; do
    [ -n "$m" ] && DEMO_MODELS+=("$m")
  done; IFS="$OLDIFS"
elif [ -f "$MODELSFILE" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && DEMO_MODELS+=("$line")
  done < "$MODELSFILE"
fi

if [ "${#DEMO_MODELS[@]}" -eq 0 ]; then
  echo "${YELLOW}No models to demo.${RESET}"
  echo "Run the setup step first:  ${BOLD}./inference-demo-setup.sh${RESET}"
  echo "(or pass MODELS=\"modelA,modelB\")"
  exit 1
fi

# ===========================================================================
# INTRO
# ===========================================================================
banner "OpenCost Inference Cost Demo"
echo "Self-hosted LLM inference runs on GPUs you pay for by the hour -- but the"
echo "bill never tells you your real cost per token, or whether input or output"
echo "is driving it. OpenCost does, straight from the vLLM metrics you already emit."
echo
echo "In this demo we'll look at ${BOLD}${#DEMO_MODELS[@]}${RESET} live model(s):"
for m in "${DEMO_MODELS[@]}"; do echo "   - $m"; done
echo
echo "For each one we'll show, live against the cluster:"
echo "  1. Allocation-basis cost: blended + input/output split + KV cache  (/inferenceCost/total?costBasis=allocation)"
echo "  2. Usage-basis cost:      same fields, active compute only          (/inferenceCost/total?costBasis=usage)"
echo "     Allocation vs. usage comparison table"
echo "  3. Hourly cost trend                                                (/inferenceCost/timeseries)"
echo
echo "First, the hardware prices OpenCost is configured with -- every cost that"
echo "follows is derived from these rates."
pause

# ===========================================================================
# CONFIGURED HARDWARE PRICING  (the basis for every cost that follows)
# ===========================================================================
banner "Configured hardware pricing"
PRICE_PATH="/pricingSourceSummary"
label "curl \"${API}${PRICE_PATH}\""
PRICE_BODY="$(apiget "$PRICE_PATH")"
echo
echo "${BOLD}Raw JSON (GPU node class -- the only class inference runs on):${RESET}"
echo "$PRICE_BODY" | jq '{ "default,gpu": .data["default,gpu"] }'
echo
echo "${BOLD}Human-readable (hourly rates per unit):${RESET}"
printf "  %-16s %14s %14s %14s\n" "NODE CLASS" "CPU (\$/vCPU)" "RAM (\$/GB)" "GPU (\$/GPU)"
echo "$PRICE_BODY" | jq -r '
  .data | to_entries[]
  | select(.key == "default,gpu")
  | [ .key,
      ((.value.CPU // "-") | if . == "" then "-" else . end),
      ((.value.RAM // "-") | if . == "" then "-" else . end),
      ((.value.GPU // "-") | if . == "" then "-" else . end)
    ] | @tsv' \
| while IFS=$'\t' read -r cls cpu ram gpu; do
    printf "  %-16s %14s %14s %14s\n" "$cls" "$cpu" "$ram" "$gpu"
  done
echo
echo "${BOLD}${YELLOW}What this means:${RESET} These are the per-hour unit rates OpenCost applies to"
echo "the GPU nodes our vLLM pods run on. The \$/GPU rate dominates inference cost --"
echo "every \$/token figure in this demo traces back to it."
pause

# ===========================================================================
# PER-MODEL API WALKTHROUGH
# ===========================================================================

# Summary accumulators (parallel arrays; bash 3.2 compatible).
# Allocation-basis fields
S_MODEL=(); S_COST=(); S_CPM=(); S_IN=(); S_OUT=(); S_RATIO=(); S_CACHE=()
# Usage-basis fields
U_COST=(); U_CPM=(); U_IN=(); U_OUT=()

LAST_BODY=""
# show_call <title> <url-path>   (uses globals HUMAN_JQ and NOTE)
show_call() {
  local title="$1" path="$2" body
  section "> ${title}"
  label "curl \"${API}${path}\""
  body="$(apiget "$path")"
  echo
  echo "${BOLD}Raw JSON:${RESET}"
  echo "$body" | jq '.'
  echo
  echo "${BOLD}Human-readable:${RESET}"
  echo "$body" | jq -r "$HUMAN_JQ"
  if [ -n "${NOTE:-}" ]; then
    echo
    echo "${BOLD}${YELLOW}What this means:${RESET} ${NOTE}"
  fi
  LAST_BODY="$body"
  pause
}

for model in "${DEMO_MODELS[@]}"; do
  banner "MODEL:  ${model}"
  enc_model=$(printf '%s' "$model" | jq -sRr @uri)

  # ---- API 1: /inferenceCost/total  (costBasis=allocation) ---------------
  HUMAN_JQ='
    .data.inferenceCosts | to_entries[0].value
    | "  Namespace ............ \(.properties.namespace)",
      "  Cost basis ........... \(.costBasis)",
      "  Total cost ........... $\(.totalCost | (.*100|round/100))",
      "  Total tokens ......... \(.totalTokens)",
      "     prompt (input) .... \(.promptTokens)",
      "     generation (out) .. \(.generationTokens)",
      "  Cost / 1M tokens ..... $\(.costPerMillionTokens | (.*100|round/100))",
      "",
      "  Input  cost .......... $\(.inputCost  | (.*100|round/100))   ( $\(.inputCostPerMillionTokens  | (.*100|round/100)) / 1M input tokens )",
      "  Output cost .......... $\(.outputCost | (.*100|round/100))   ( $\(.outputCostPerMillionTokens | (.*100|round/100)) / 1M output tokens )",
      "  Output:Input ratio ... \((.outputCostPerMillionTokens / (if .inputCostPerMillionTokens>0 then .inputCostPerMillionTokens else 1 end)) | (.*100|round/100))x",
      "  Allocation method .... \(.allocationMethod)",
      "  KV cache savings ..... \((.cacheSavingsFraction*100) | (.*10|round/10))% of prompt tokens served from cache"'
  NOTE="Allocation basis includes idle GPU time and shared infra -- it reconciles to the infrastructure bill. OpenCost splits the GPU cost into input vs. output using vLLM's actual prefill/decode timing (allocationMethod=compute_time). cacheSavingsFraction shows how much prefill work the KV prefix cache avoided."
  show_call "API 1/3 -- /inferenceCost/total  (costBasis=allocation)" \
    "/inferenceCost/total?window=${WINDOW}&aggregate=model_name,namespace&filter=model_name:${enc_model}&costBasis=allocation"
  read -r c_cost c_cpm c_in c_out c_ratio c_cache < <(echo "$LAST_BODY" | jq -r '
    .data.inferenceCosts | to_entries[0].value
    | "\(.totalCost) \(.costPerMillionTokens) \(.inputCostPerMillionTokens) \(.outputCostPerMillionTokens) \((.outputCostPerMillionTokens/(if .inputCostPerMillionTokens>0 then .inputCostPerMillionTokens else 1 end))) \(.cacheSavingsFraction)"')

  # ---- API 2: /inferenceCost/total  (costBasis=usage) --------------------
  HUMAN_JQ='
    .data.inferenceCosts | to_entries[0].value
    | "  Namespace ............ \(.properties.namespace)",
      "  Cost basis ........... \(.costBasis)",
      "  Total cost ........... $\(.totalCost | (.*100|round/100))",
      "  Total tokens ......... \(.totalTokens)",
      "     prompt (input) .... \(.promptTokens)",
      "     generation (out) .. \(.generationTokens)",
      "  Cost / 1M tokens ..... $\(.costPerMillionTokens | (.*100|round/100))",
      "",
      "  Input  cost .......... $\(.inputCost  | (.*100|round/100))   ( $\(.inputCostPerMillionTokens  | (.*100|round/100)) / 1M input tokens )",
      "  Output cost .......... $\(.outputCost | (.*100|round/100))   ( $\(.outputCostPerMillionTokens | (.*100|round/100)) / 1M output tokens )",
      "  Output:Input ratio ... \((.outputCostPerMillionTokens / (if .inputCostPerMillionTokens>0 then .inputCostPerMillionTokens else 1 end)) | (.*100|round/100))x",
      "  Allocation method .... \(.allocationMethod)",
      "  KV cache savings ..... \((.cacheSavingsFraction*100) | (.*10|round/10))% of prompt tokens served from cache"'
  NOTE="Usage basis reflects active compute only -- idle GPU time and shared infra costs are excluded. This is the efficiency view: what did the model actually consume? A large gap vs. allocation means significant idle GPU capacity."
  show_call "API 2/3 -- /inferenceCost/total  (costBasis=usage)" \
    "/inferenceCost/total?window=${WINDOW}&aggregate=model_name,namespace&filter=model_name:${enc_model}&costBasis=usage"
  read -r u_cost u_cpm u_in u_out < <(echo "$LAST_BODY" | jq -r '
    .data.inferenceCosts | to_entries[0].value
    | "\(.totalCost) \(.costPerMillionTokens) \(.inputCostPerMillionTokens) \(.outputCostPerMillionTokens)"')

  # ---- Allocation vs. usage comparison table -----------------------------
  section "> Allocation vs. Usage comparison"
  idle_pct=$(awk "BEGIN{
    a=${c_cost:-0}; u=${u_cost:-0}
    if (a>0) printf \"%.1f\", (1-(u/a))*100
    else     printf \"0.0\"
  }")
  printf "\n  %-22s %12s %12s %12s\n" "" "ALLOCATION" "USAGE" "IDLE %"
  printf "  %-22s %12s %12s %12s\n" "$(printf '%0.s-' {1..22})" "$(printf '%0.s-' {1..12})" "$(printf '%0.s-' {1..12})" "$(printf '%0.s-' {1..12})"
  printf "  %-22s %12.2f %12.2f %11s%%\n" "Total cost (\$)"      "${c_cost:-0}"  "${u_cost:-0}"  "${idle_pct}"
  printf "  %-22s %12.2f %12.2f\n"         "Cost/1M tokens (\$)"  "${c_cpm:-0}"   "${u_cpm:-0}"
  printf "  %-22s %12.2f %12.2f\n"         "Input \$/1M"          "${c_in:-0}"    "${u_in:-0}"
  printf "  %-22s %12.2f %12.2f\n"         "Output \$/1M"         "${c_out:-0}"   "${u_out:-0}"
  echo
  echo "${BOLD}${YELLOW}What this means:${RESET} Allocation reconciles to your bill (idle + shared infra included)."
  echo "Usage reflects only active compute. The idle % is GPU capacity you reserved"
  echo "but didn't use -- a direct signal for right-sizing or bin-packing opportunities."
  pause

  # ---- API 3: /inferenceCost/timeseries ----------------------------------
  HUMAN_JQ="
    .data.inferenceCostSets
    | map({ hour: (.window.start), cost: ((.inferenceCosts[\"${model}\"].totalCost // 0)) })
    | map(select(.cost > 0))
    | (\"  hourly cost trend (\\(length) active hours):\"),
      (.[] | \"    \\(.hour)   \$\\(.cost | (.*100|round/100))\")"
  NOTE="The same numbers OpenCost exports as Prometheus gauges (llm_total_hourly_cost, llm_cost_per_million_tokens, llm_cache_savings_fraction) -- feed straight into Grafana dashboards and alerts. Cost tracks load hour by hour."
  show_call "API 3/3 -- /inferenceCost/timeseries  (accumulate=${ACCUMULATE})" \
    "/inferenceCost/timeseries?window=${WINDOW}&accumulate=${ACCUMULATE}&aggregate=model_name&filter=model_name:${enc_model}"

  S_MODEL+=("$model"); S_COST+=("$c_cost"); S_CPM+=("$c_cpm")
  S_IN+=("$c_in"); S_OUT+=("$c_out"); S_RATIO+=("$c_ratio"); S_CACHE+=("$c_cache")
  U_COST+=("$u_cost"); U_CPM+=("$u_cpm"); U_IN+=("$u_in"); U_OUT+=("$u_out")
done

# ===========================================================================
# OVERALL SUMMARY
# ===========================================================================
banner "Overall summary"

echo "${BOLD}Allocation basis  (reconciles to bill — idle + shared infra included):${RESET}"
echo
printf "  %-38s %10s %10s %10s %10s %8s %8s\n" "MODEL" "COST(\$)" "\$/1M" "IN \$/1M" "OUT \$/1M" "OUT:IN" "CACHE%"
i=0
while [ "$i" -lt "${#S_MODEL[@]}" ]; do
  cache_pct=$(awk "BEGIN{printf \"%.0f\", ${S_CACHE[$i]:-0}*100}")
  printf "  %-38s %10.2f %10.2f %10.2f %10.2f %7.1fx %7s%%\n" \
    "${S_MODEL[$i]:0:38}" "${S_COST[$i]:-0}" "${S_CPM[$i]:-0}" \
    "${S_IN[$i]:-0}" "${S_OUT[$i]:-0}" "${S_RATIO[$i]:-0}" "$cache_pct"
  i=$((i+1))
done

echo
echo "${BOLD}Usage basis  (active compute only — idle excluded):${RESET}"
echo
printf "  %-38s %10s %10s %10s %10s %8s\n" "MODEL" "COST(\$)" "\$/1M" "IN \$/1M" "OUT \$/1M" "IDLE %"
i=0
while [ "$i" -lt "${#S_MODEL[@]}" ]; do
  idle_pct=$(awk "BEGIN{
    a=${S_COST[$i]:-0}; u=${U_COST[$i]:-0}
    if (a>0) printf \"%.1f\", (1-(u/a))*100
    else     printf \"0.0\"
  }")
  printf "  %-38s %10.2f %10.2f %10.2f %10.2f %7s%%\n" \
    "${S_MODEL[$i]:0:38}" "${U_COST[$i]:-0}" "${U_CPM[$i]:-0}" \
    "${U_IN[$i]:-0}" "${U_OUT[$i]:-0}" "${idle_pct}"
  i=$((i+1))
done

echo
echo "${BOLD}${YELLOW}Key takeaways${RESET}"
cat <<'EOF'
  1. COST PER TOKEN IS REAL, NOT LIST PRICE.
     Every figure above is your actual GPU/infra bill divided by the tokens
     vLLM processed -- measured, per model, over a real time window.

  2. ALLOCATION vs USAGE REVEALS IDLE GPU.
     Allocation reconciles to your bill (idle + shared infra included).
     Usage reflects only active compute. The gap is idle GPU capacity --
     a direct signal for right-sizing or bin-packing opportunities.

  3. INPUT vs OUTPUT IS MEASURED, NOT ASSUMED.
     The split comes from vLLM's own prefill/decode timing (compute_time),
     so each model shows its own output:input ratio. Prefill-heavy workloads
     (lots of prompt, little generation) look very different from chat-style
     ones -- and OpenCost captures that difference instead of guessing.

  4. KV CACHE SAVINGS ARE VISIBLE.
     cacheSavingsFraction shows how much prompt-side compute the prefix cache
     avoided -- a direct, tunable cost lever unique to self-hosted inference.

  5. IT'S ALL PROMETHEUS-NATIVE.
     Totals, timeseries, and Grafana gauges come from the standard vLLM
     metrics you already emit -- no new instrumentation required.
EOF
echo
hr
echo "${BOLD}${GREEN}Demo complete.${RESET}"
hr
