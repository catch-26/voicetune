#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }
have() { command -v "$1" >/dev/null; }
need() { declare -p "$1" >/dev/null 2>&1 && return; local v=$2; [ -z "${3:-}" ] || v=$(ask "$3" "$2"); printf '%s=%s\n' "$1" "$v" >> config.env; export "$1=$v"; }
KSHA=a12d0410e89841e6f3c09958ae5c072f90ae1d49 SSHA=b1956da84bf4a6ccc88f2440078024f1c4bfec7d OSHA=b96fef95e6a746495f92443fac7c688f90fc57fc
RUN=${1:-}
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
[ -n "$RUN" ] || RUN=$(ask "run which workflows: stt | tts | both" "${RUN_LAST:-both}")
case $RUN in stt|tts|both) ;; *) echo "unknown workflow '$RUN': use stt, tts or both"; exit 1 ;; esac
grep -q '^RUN_LAST=' config.env && sed -i "s/^RUN_LAST=.*/RUN_LAST=$RUN/" config.env || echo "RUN_LAST=$RUN" >> config.env
[ "$RUN" = both ] && RUN=stt,tts; export RUN
if [[ $RUN == *tts* ]]; then
  [[ ",$LANGS," == *,de,* ]] || { echo "the TTS workflow is German: add de to LANGS in config.env (and de_de to FLEURS_CFGS)"; exit 1; }
  need TTS_CLIPS 600 "German sentences to record for TTS in total (~5.5 s each; the published kikiri voices used 455-627)"
  need TTS_VOICE myvoice "voice name for the TTS output files"
  need TTS_BASE kikiri-tts/kikiri-german-base-51speakers-synthetic/kikiri_german_base_51spk_ep4.pth "TTS base weights: German kikiri base (default) | hexgrad/Kokoro-82M/kokoro-v1_0.pth (English base) | a local .pth"
  need TTS_GATE_MODEL openai/whisper-large-v3-turbo; need TTS_GATE_WER 0.15; need TTS_EPOCHS_1 10; need TTS_EPOCHS_2 10; need TTS_BATCH 4
fi
T="$PWD/.tools"
export HF_HOME="$DOWNLOAD_DIR/hf" PATH="$T/bin:$PATH" UV_TOOL_DIR="$T/uv" UV_TOOL_BIN_DIR="$T/bin"
mkdir -p data/wav models "$OUT_DIR" "$DOWNLOAD_DIR" "$T/bin"
have uv || { echo "uv not found, installing it into $T/bin"; python3 fetch.py "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" "$T/bin"; }
venv() { export VIRTUAL_ENV="$PWD/$1"; [ -d "$1" ] || uv venv --python "$2" "$1"; if [ ! -f "$1/.installed" ]; then uv pip install --torch-backend "$TORCH_BACKEND" -r "$3"; touch "$1/.installed"; fi; }
if [[ $RUN == *tts* ]]; then
  KIKIRI="$DOWNLOAD_DIR/third_party/kikiri-tts-${KSHA:0:7}"; export KIKIRI
  if [ ! -f "$KIKIRI/.ok" ]; then
    rm -rf "$KIKIRI"; G=https://github.com/semidark
    python3 fetch.py "$G/kikiri-tts/archive/$KSHA.tar.gz" "$KIKIRI"; python3 fetch.py "$G/StyleTTS2/archive/$SSHA.tar.gz" "$KIKIRI/StyleTTS2"; python3 fetch.py "$G/kokoro/archive/$OSHA.tar.gz" "$KIKIRI/kokoro"
    touch "$KIKIRI/.ok"
  fi
  venv .venv-tts 3.12 requirements-tts.txt; PYT=.venv-tts/bin/python
  export PYTHONPATH="$KIKIRI/kokoro:$KIKIRI/StyleTTS2"
  EL=$($PYT -c 'import espeakng_loader, os; print(os.path.dirname(espeakng_loader.__file__))')
  if have espeak-ng && [ ! -L "$EL/libespeak-ng.so" ]; then
    LDC=$(command -v ldconfig || echo /sbin/ldconfig); L=$($LDC -p | grep -o '/[^ ]*libespeak-ng\.so\.1$' | head -1); ED=$(espeak-ng --version | sed -n 's/.*Data at: *//p')
    if [ -n "$L" ] && [ -d "$ED" ]; then echo "using the system espeak-ng ($L)"; rm -rf "$EL/espeak-ng-data" "$EL/libespeak-ng.so"; ln -s "$L" "$EL/libespeak-ng.so"; ln -s "$ED" "$EL/espeak-ng-data"; fi
  fi
  $PYT -c "from misaki import de; assert de.DEG2P()('Guten Tag')[0]" || { echo "German phonemizer failed: install espeak-ng with your package manager and rerun"; exit 1; }
fi
if [[ $RUN == *stt* ]]; then venv ".venv-$ENGINE" "$([ "$ENGINE" = parakeet ] && echo 3.12 || echo 3.13)" "requirements-$ENGINE.txt"; PY=.venv-$ENGINE/bin/python; PYD=$PY; else PYD=$PYT; fi
VIRTUAL_ENV="$PWD/$(dirname "$(dirname "$PYD")")"; export VIRTUAL_ENV
if [ "$TORCH_BACKEND" != cpu ] && [ "${DEVICE:-}" != cpu ] && ! grep -q '^GPU=' config.env; then
  l=$($PYD -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"); GPU=
  if [ "$(printf '%s\n' "$l" | grep -c .)" -gt 1 ]; then echo "$l"; GPU=$(ask "several GPUs found: index to train on (saved as GPU= in config.env)" 0); fi
  echo "GPU=$GPU" >> config.env
fi
[ -z "${GPU:-}" ] || export HIP_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU"
PY=${PY:-$PYD}
if ! have ffmpeg; then
  echo "ffmpeg not found, installing static-ffmpeg as a uv tool"; uv tool install static-ffmpeg
  F=$(static_ffmpeg_paths | sed -n 's/^FFMPEG=//p'); [ -x "$F" ] || { echo "static-ffmpeg download failed: install ffmpeg with your package manager and rerun"; exit 1; }
  ln -sf "$F" "$T/bin/ffmpeg"
fi
have openssl || $PYD -c "import cryptography" 2>/dev/null || uv pip install cryptography
$PYD -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, using CPU (slow but works)')"
if [ "$TORCH_BACKEND" != cpu ] && ! $PYD -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU not visible to torch. checks: torch version above must end in +$TORCH_BACKEND (else: rm -rf $VIRTUAL_ENV and rerun);"
  echo "  ls -l /dev/kfd /dev/dri (you need read/write: add yourself to the video and render groups, re-login);"
  echo "  for RX 7700/7800/7600 export HSA_OVERRIDE_GFX_VERSION=11.0.0 and rerun."
  read -r -p "continue on CPU anyway? [y/N]: " c; [ "$c" = y ] || exit 1
fi
hf_get() { if have hf; then hf download "$@" >/dev/null; else "$VIRTUAL_ENV/bin/hf" download "$@" >/dev/null; fi; }
if [[ $RUN == *stt* ]]; then if [ "$ENGINE" = parakeet ]; then hf_get "$PARAKEET_MODEL"; else hf_get "openai/whisper-$WHISPER_MODEL"; fi; fi
[ -f data/prompts.txt ] || $PYD stage0_data.py
TARGET=$N_PROMPTS
if [[ $RUN == *tts* ]]; then $PYT tts_prompts.py; TARGET=$(wc -l < data/prompts.txt); fi
n=$( [ -f data/manifest.jsonl ] && wc -l < data/manifest.jsonl || echo 0 )
if [ "$TARGET" -gt 0 ] && [ "$n" -lt "$TARGET" ]; then
  echo "recorded $n/$TARGET — starting recorder"
  N_PROMPTS=$TARGET $PYD stage1_record.py
fi
DONE=
if [[ $RUN == *tts* ]]; then
  TTS_OUT="$OUT_DIR/kokoro-de-$TTS_VOICE.pth"
  if [ ! -f "$TTS_OUT" ]; then $PYT tts_prep.py; $PYT tts_train.py; fi
  DONE="$TTS_OUT $OUT_DIR/$TTS_VOICE.pt"
fi
if [[ $RUN != *stt* ]]; then echo "done: $DONE"; exit 0; fi
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
echo "done: $(compgen -G "$OUT_GLOB") $DONE"
