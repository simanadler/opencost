#!/usr/bin/env bash
#
# OpenCost Inference Cost Demo — SETUP (run privately BEFORE the demo)
# --------------------------------------------------------------------
# Probes the OpenCost inference API, finds every model with clean cost
# attribution (nonzero cost AND nonzero tokens AND a real allocationMethod),
# then picks the TWO most CONTRASTING models to make the best live demo:
#
#   Pick #1: highest token volume        -> a busy, high-throughput workload
#   Pick #2: highest output:input ratio  -> a prefill-heavy / different-shape
#            (from the remaining models)     workload that looks very different
#
# The chosen models are written to a small file (default: .demo-models) that
# the live script (inference-demo.sh) reads. This setup step is the ONLY place
# that ever inspects or reports skipped / mislabeled models — the live demo
# never sees them.
#
# Requirements: bash, curl, jq. OpenCost API reachable at $API (port-forward 9003).
#
# Usage:
#   ./inference-demo-setup.sh                 # 24h window, auto-pick 2, write .demo-models
#   WINDOW=6h ./inference-demo-setup.sh
#   OUTFILE=/tmp/demo-models ./inference-demo-setup.sh
#
set -uo pipefail

API="${API:-http://localhost:9003}"
WINDOW="${WINDOW:-24h}"
CURL_TIMEOUT="${CURL_TIMEOUT:-120}"
OUTFILE="${OUTFILE:-.demo-models}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; CYAN=""
fi
DASH="----------------------------------------------------------------------"

command -v jq   >/dev/null 2>&1 || { echo "ERROR: jq not found";   exit 1; }
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }

echo "$DASH"
echo "${BOLD}${CYAN}Inference Demo Setup${RESET}  |  API: ${API}  |  Window: ${WINDOW}"
echo "$DASH"
echo "This runs BEFORE the demo (not shown live). It selects the best models"
echo "and writes them to: ${BOLD}${OUTFILE}${RESET}"
echo "Make sure you did: oc port-forward -n opencost svc/opencost 9003:9003"
echo

echo "Probing ${API}/inferenceCost/total?window=${WINDOW}&aggregate=model_name,namespace ..."
RAW="$(curl -s --max-time "$CURL_TIMEOUT" "${API}/inferenceCost/total?window=${WINDOW}&aggregate=model_name,namespace")"

if [ -z "$RAW" ] || [ "$(echo "$RAW" | jq -r '.data.inferenceCosts | length' 2>/dev/null)" = "0" ]; then
  echo "${YELLOW}No inference data returned. Check the port-forward and window.${RESET}"
  exit 1
fi

# ---- Diagnostics (private): full model inventory --------------------------
echo
echo "${BOLD}Full inventory (setup-only diagnostics):${RESET}"
printf "  %-42s %-20s %12s %16s %-14s %s\n" "MODEL" "NAMESPACE" "COST(\$)" "TOKENS" "METHOD" "STATUS"
while IFS=$'\t' read -r m ns c t meth st; do
  [ -z "$m" ] && continue
  cfmt=$(printf "%.2f" "$c" 2>/dev/null || echo "$c")
  [ -z "$meth" ] && meth="-"
  if [ "$st" = "OK" ]; then col="$GREEN"; else col="$DIM"; fi
  printf "  ${col}%-42s %-20s %12s %16s %-14s %s${RESET}\n" \
    "${m:0:42}" "${ns:0:20}" "$cfmt" "$t" "$meth" "$st"
done < <(echo "$RAW" | jq -r '
  .data.inferenceCosts | to_entries[] | .value as $v
  | ($v.totalCost) as $c
  | ($v.totalTokens) as $t
  | ($v.allocationMethod // "") as $meth
  | (if ($c > 0 and $t > 0 and $meth != "") then "OK" else "SKIP" end) as $st
  | (if (($v.properties.namespace // "") == "") then "-" else $v.properties.namespace end) as $ns
  | [$v.properties.modelName, $ns,
     ($c|tostring), ($t|tostring), $meth, $st] | @tsv')

# ---- Selection: most-contrasting pair -------------------------------------
# Demo-ready rows as TSV: modelName \t tokens \t ratio  (ratio = out$/in$).
READY="$(echo "$RAW" | jq -r '
  .data.inferenceCosts | to_entries[] | .value
  | select(.totalCost > 0 and .totalTokens > 0 and (.allocationMethod // "") != "")
  | [ .properties.modelName,
      (.totalTokens|tostring),
      ((.outputCostPerMillionTokens / (if .inputCostPerMillionTokens>0 then .inputCostPerMillionTokens else 1 end))|tostring)
    ] | @tsv' | sort -u)"

READY_COUNT=$(printf '%s\n' "$READY" | sed '/^$/d' | wc -l | tr -d ' ')

echo
echo "${BOLD}Demo-ready models: ${READY_COUNT}${RESET}"

if [ "$READY_COUNT" -eq 0 ]; then
  echo "${YELLOW}No demo-ready models (nonzero cost + tokens + real method). Fix pod labeling or widen WINDOW.${RESET}"
  exit 1
fi

# Pick #1 = highest token volume.
PICK1="$(printf '%s\n' "$READY" | sort -t$'\t' -k2,2 -nr | head -1 | cut -f1)"
# Pick #2 = highest output:input ratio among the rest.
PICK2="$(printf '%s\n' "$READY" | grep -v -F "$(printf '%s\t' "$PICK1")" \
          | sort -t$'\t' -k3,3 -nr | head -1 | cut -f1)"

# Assemble final list (dedup, drop empties), preserving pick order.
CHOSEN=()
for p in "$PICK1" "$PICK2"; do
  [ -z "$p" ] && continue
  dup=0
  for e in "${CHOSEN[@]:-}"; do [ "$e" = "$p" ] && dup=1; done
  [ "$dup" -eq 0 ] && CHOSEN+=("$p")
done

if [ "${#CHOSEN[@]}" -eq 0 ]; then
  echo "${YELLOW}Selection failed unexpectedly.${RESET}"; exit 1
fi

# ---- Write the output file ------------------------------------------------
: > "$OUTFILE"
for m in "${CHOSEN[@]}"; do echo "$m" >> "$OUTFILE"; done

echo
echo "${BOLD}${GREEN}Selected ${#CHOSEN[@]} model(s) for the demo:${RESET}"
n=1
for m in "${CHOSEN[@]}"; do
  if [ "$n" -eq 1 ]; then why="(highest token volume)"; else why="(most contrasting cost shape)"; fi
  echo "   ${n}. ${m}  ${DIM}${why}${RESET}"
  n=$((n+1))
done
echo
echo "Written to: ${BOLD}${OUTFILE}${RESET}"
echo "${DIM}Run the live demo with:  ./inference-demo.sh${RESET}"
