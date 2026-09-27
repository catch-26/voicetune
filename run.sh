#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }
have() { command -v "$1" >/dev/null; }
if [ ! -f config.env ]; then
  echo "first run: answer or press enter for the default"
  TORCH_BACKEND=$(ask "torch build: rocm7.2 (AMD), cpu, cu128 (NVIDIA), or auto" rocm7.2)
  ENGINE=$(ask "engine: parakeet (-> Offline Voice Input, GGUF) | whisper (-> FUTO Keyboard, GGML)" parakeet)
  WHISPER_MODEL=; PARAKEET_MODEL=
  if [ "$ENGINE" = parakeet ]; then
    PARAKEET_MODEL=$(ask "parakeet model: nvidia/parakeet-tdt-0.6b-v3 (25 languages) | nvidia/parakeet-tdt-0.6b-v2, nvidia/parakeet-tdt_ctc-110m, nvidia/parakeet-ctc-0.6b (English only)" nvidia/parakeet-tdt-0.6b-v3)
  else
    WHISPER_MODEL=$(ask "whisper size: tiny.en (39M, English only) | tiny (39M) | base (74M) | small (244M)" base)
  fi
  REPLAY_CFGS=; REPLAY_N=0; EN_FIXED=
  if [ "$WHISPER_MODEL" = tiny.en ] || { [ "$ENGINE" = parakeet ] && [[ "$PARAKEET_MODEL" != *-v3* ]]; }; then EN_FIXED=1; LANGS=en; FLEURS_CFGS=en_us; echo "English-only model: language fixed to en"
  else
    LANGS=$(ask "languages you will record, ISO codes, comma separated" "en,de")
    FLEURS_CFGS=$(ask "matching FLEURS configs (huggingface.co/datasets/google/fleurs)" "en_us,de_de")
  fi
  if [ -z "$EN_FIXED" ]; then
    REPLAY_CFGS=$(ask "other-language FLEURS configs to protect from forgetting (empty = none)" "es_419,it_it,ru_ru")
    REPLAY_N=$(ask "clips per protected language (half train, half check)" 150)
  fi
  N_PROMPTS=$(ask "sentences to record in total (~5.5 s each; 400 = 35 min, 650 = 1 h)" 400)
  MIX_PUBLIC=$(ask "FLEURS audio of your languages in the finetune: none | anchor (~200 clips/lang, keeps other speakers good) | full (10 h/lang, retune the base model, no recordings needed)" anchor)
  ACFT_PUBLIC=y; if [ "$ENGINE" = whisper ]; then ACFT_PUBLIC=$(ask "use FLEURS for the ACFT step (recommended)" y); fi
  DOWNLOAD_DIR=$(ask "download folder" "$PWD/downloads")
  OUT_DIR=$(ask "output folder for the finished model" "$PWD/out")
  printf 'ENGINE=%s\nPARAKEET_MODEL=%s\nWHISPER_MODEL=%s\nANCHOR_N=200\nANCHOR_EVAL=100\nTORCH_BACKEND=%s\nLANGS=%s\nFLEURS_CFGS=%s\nREPLAY_CFGS=%s\nREPLAY_N=%s\nN_PROMPTS=%s\nMIX_PUBLIC=%s\nACFT_PUBLIC=%s\nDOWNLOAD_DIR=%s\nOUT_DIR=%s\nHOLDOUT=0.15\nPORT=8443\nFT_EPOCHS=3\nACFT_EPOCHS=8\nFORGET_TOL=0.10\n' \
    "$ENGINE" "$PARAKEET_MODEL" "$WHISPER_MODEL" "$TORCH_BACKEND" "$LANGS" "$FLEURS_CFGS" "$REPLAY_CFGS" "$REPLAY_N" "$N_PROMPTS" "$MIX_PUBLIC" "$ACFT_PUBLIC" "$DOWNLOAD_DIR" "$OUT_DIR" > config.env
  echo "saved to config.env (edit or delete to redo)"
fi
set -a; source config.env; ENGINE=${ENGINE:-whisper}; WHISPER_MODEL=${WHISPER_MODEL:-base}; set +a
T="$PWD/.tools"
export HF_HOME="$DOWNLOAD_DIR/hf" PATH="$T/bin:$PATH" UV_TOOL_DIR="$T/uv" UV_TOOL_BIN_DIR="$T/bin"
mkdir -p data/wav models "$OUT_DIR" "$DOWNLOAD_DIR" "$T/bin"
have uv || { echo "uv not found, installing it into $T/bin"; python3 fetch.py "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" "$T/bin"; }
V=.venv-$ENGINE; export VIRTUAL_ENV="$PWD/$V"
[ -d "$V" ] || uv venv --python "$([ "$ENGINE" = parakeet ] && echo 3.12 || echo 3.13)" "$V"
if [ ! -f "$V/.installed" ]; then uv pip install --torch-backend "$TORCH_BACKEND" -r "requirements-$ENGINE.txt"; touch "$V/.installed"; fi
PY=$V/bin/python
have ffmpeg || { echo "ffmpeg not found, using the imageio-ffmpeg build"; uv pip install imageio-ffmpeg; ln -sf "$($PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$T/bin/ffmpeg"; }
have openssl || $PY -c "import cryptography" 2>/dev/null || uv pip install cryptography
$PY -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, using CPU (slow but works)')"
if [ "$TORCH_BACKEND" != cpu ] && ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU not visible to torch. checks: torch version above must end in +$TORCH_BACKEND (else: rm -rf $V and rerun);"
  echo "  ls -l /dev/kfd /dev/dri (you need read/write: add yourself to the video and render groups, re-login);"
  echo "  for RX 7700/7800/7600 export HSA_OVERRIDE_GFX_VERSION=11.0.0 and rerun."
  read -r -p "continue on CPU anyway? [y/N]: " c; [ "$c" = y ] || exit 1
fi
hf_get() { if have hf; then hf download "$@" >/dev/null; else "$V/bin/hf" download "$@" >/dev/null; fi; }
if [ "$ENGINE" = parakeet ]; then hf_get "$PARAKEET_MODEL"; else hf_get "openai/whisper-$WHISPER_MODEL"; fi
[ -f data/prompts.txt ] || $PY stage0_data.py
n=$( [ -f data/manifest.jsonl ] && wc -l < data/manifest.jsonl || echo 0 )
if [ "$N_PROMPTS" -gt 0 ] && [ "$n" -lt "$N_PROMPTS" ]; then
  echo "recorded $n/$N_PROMPTS — starting recorder"
  $PY stage1_record.py
fi
if [ "$ENGINE" = parakeet ]; then
  [ -f models/parakeet/ft.nemo ] || $PY stage2_parakeet.py
  OUT_GLOB="$OUT_DIR/$(basename "$PARAKEET_MODEL")-*-ft-*.gguf"
else
  [ -f models/ft/config.json ] || $PY stage2_whisper.py
  [ -f models/acft/config.json ] || $PY stage3_whisper_acft.py
  OUT_GLOB="$OUT_DIR/*q8_0.bin"
fi
if ! compgen -G "$OUT_GLOB" >/dev/null; then
  have cmake || uv tool install cmake
  have make || { have ninja || uv tool install ninja; export CMAKE_GENERATOR=Ninja; }
  if [ -x "$T/bin/c++" ] || ! { have c++ || have g++ || have clang++; }; then
    have python-zig || uv tool install ziglang
    for c in cc c++ ar ranlib; do
      if [ "$c" = cc ] || [ "$c" = c++ ] || ! have "$c"; then printf '#!/bin/sh\nexec python-zig %s "$@"\n' "$c" > "$T/bin/$c"; chmod +x "$T/bin/$c"; fi
    done
    export CC="$T/bin/cc" CXX="$T/bin/c++"
  fi
  if [ "$ENGINE" = parakeet ]; then PY=$PY bash stage4_parakeet.sh; else PY=$PY bash stage4_whisper.sh; fi
fi
echo "done: $(compgen -G "$OUT_GLOB")"
