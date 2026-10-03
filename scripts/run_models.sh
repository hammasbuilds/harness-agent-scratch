#!/usr/bin/env bash
# The model arm, end to end: every model in MODELS over the task suite.
#
#   bash scripts/run_models.sh --dry-run   # the job list and call count; needs no model
#   bash scripts/run_models.sh             # the real run (GPU, Ollama)
#
# Before a real run it checks free RAM, that a GPU with enough free memory has no
# other compute process on it, and that Ollama answers with every model pulled.
# Generations are cached in .bench-cache/, so a rerun after an interruption only
# pays for the calls it has not made yet. Results: results/model_runs/<model>.json.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=${MODELS:-"qwen2.5:7b-instruct qwen2.5:14b-instruct qwen2.5-coder:14b"}
OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-8}
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-14000}   # qwen2.5 14B at Q4 with an 8k context
NUM_CTX=${NUM_CTX:-8192}
MAX_STEPS=${MAX_STEPS:-20}

model_args=()
for m in $MODELS; do model_args+=(--model "$m"); done
if [[ -n "${PYTHON:-}" ]]; then   # one interpreter path, which may contain spaces
    python=("$PYTHON")
else
    python=(uv run python)
fi
bench=("${python[@]}" -m harness.bench "${model_args[@]}" --num-ctx "$NUM_CTX" --max-steps "$MAX_STEPS")

if [[ "${1:-}" == "--dry-run" ]]; then
    "${bench[@]}" --dry-run
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "usage: bash scripts/run_models.sh [--dry-run]" >&2
    exit 2
fi

free_ram_gb() {
    if [[ -r /proc/meminfo ]]; then
        awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo
    else
        powershell -NoProfile -Command \
            "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'
    fi
}

ram=$(free_ram_gb)
if (( ram < MIN_FREE_RAM_GB )); then
    echo "only ${ram} GB of RAM free (need ${MIN_FREE_RAM_GB}); not starting" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found; this run is meant for the GPU" >&2
    exit 1
fi
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vi ollama || true)
if [[ -n "$busy" ]]; then
    echo "the GPU is in use by another process; not starting:" >&2
    echo "$busy" >&2
    exit 1
fi
vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' \r')
if (( vram < MIN_FREE_VRAM_MB )); then
    echo "only ${vram} MB of GPU memory free (need ${MIN_FREE_VRAM_MB}); not starting" >&2
    exit 1
fi

tags=$(curl -sf --max-time 10 "$OLLAMA_URL/api/tags") || {
    echo "Ollama does not answer at $OLLAMA_URL" >&2
    exit 1
}
for m in $MODELS; do
    if ! grep -q "\"name\":\"$m\"" <<<"$tags"; then
        echo "model $m is not pulled (ollama pull $m)" >&2
        exit 1
    fi
done

echo "free RAM ${ram} GB, free VRAM ${vram} MB; running: $MODELS"
"${bench[@]}" --num-gpu 99 --base-url "$OLLAMA_URL"
