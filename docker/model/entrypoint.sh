#!/usr/bin/env bash
# =============================================================================
# Model container entrypoint — llama-server (Q8 only)
# =============================================================================
# GGUFs are expected at $GGUF_DIR (bind-mounted from host on first deploy,
# or auto-downloaded from HuggingFace if missing).
#
# IMPORTANT: Only clinicdx-v1-q8.gguf is used. No Q4 fallback exists.
# CPU mode uses the exact same Q8 file with N_GPU_LAYERS=0.
# =============================================================================
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/gguf_data}"
HF_MODEL_REPO="${HF_MODEL_REPO:-ClinicDx1/ClinicDx}"
MODEL_PORT="${MODEL_PORT:-8180}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
MODEL_CTX="${MODEL_CTX:-8192}"
MODEL_PARALLEL="${MODEL_PARALLEL:-4}"
MODEL_THREADS="${MODEL_THREADS:-8}"

MODEL_FILE="${GGUF_DIR}/clinicdx-v1-q8.gguf"
ENCODER_FILE="${GGUF_DIR}/medasr-encoder.gguf"
AUDIO_PROJ_FILE="${GGUF_DIR}/audio-projector-v3-best.gguf"

log() {
  printf '{"ts":"%s","level":"INFO","service":"model-entrypoint","msg":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}
err() {
  printf '{"ts":"%s","level":"ERROR","service":"model-entrypoint","msg":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

mkdir -p "${GGUF_DIR}"

# ── Download only if missing (skipped when bind-mounted from host) ────────────
download_if_missing() {
  local filename="$1"
  local dest="${GGUF_DIR}/${filename}"
  if [ ! -f "${dest}" ]; then
    log "Downloading ${filename} — not found at ${dest}"
    if [ -n "${HF_TOKEN:-}" ]; then
      huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
    fi
    huggingface-cli download "${HF_MODEL_REPO}" \
      "${filename}" \
      --local-dir "${GGUF_DIR}" \
      --local-dir-use-symlinks False
    if [ ! -f "${dest}" ]; then
      err "Download failed: ${filename} still missing at ${dest}"
      exit 1
    fi
    log "Download complete — ${filename} ($(du -sh "${dest}" | cut -f1))"
  else
    log "Found ${filename} ($(du -sh "${dest}" | cut -f1))"
  fi
}

# Q8 only — no Q4 check, no Q4 download, no fallback
download_if_missing "clinicdx-v1-q8.gguf"
download_if_missing "medasr-encoder.gguf"
download_if_missing "audio-projector-v3-best.gguf"

# ── Log launch config ─────────────────────────────────────────────────────────
printf '{"ts":"%s","level":"INFO","service":"model-entrypoint","msg":"launching llama-server","model":"%s","n_gpu_layers":%s,"ctx":%s,"parallel":%s,"port":%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${MODEL_FILE}" "${N_GPU_LAYERS}" "${MODEL_CTX}" "${MODEL_PARALLEL}" "${MODEL_PORT}"

# ── Launch llama-server ───────────────────────────────────────────────────────
exec llama-server \
  --model        "${MODEL_FILE}" \
  --medasr-encoder "${ENCODER_FILE}" \
  --audio-proj   "${AUDIO_PROJ_FILE}" \
  --n-gpu-layers "${N_GPU_LAYERS}" \
  --ctx-size     "${MODEL_CTX}" \
  --parallel     "${MODEL_PARALLEL}" \
  --threads      "${MODEL_THREADS}" \
  --host         0.0.0.0 \
  --port         "${MODEL_PORT}" \
  --log-disable
