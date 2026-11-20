#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }
have() { command -v "$1" >/dev/null; }
kv() { if [[ $2 =~ ^[[:alnum:]_.,/:+@-]*$ ]]; then printf '%s=%s\n' "$1" "$2"; else printf '%s=%q\n' "$1" "$2"; fi; }
need() { declare -p "$1" >/dev/null 2>&1 && return; local v=$2; [ -z "${3:-}" ] || v=$(ask "$3" "$2"); kv "$1" "$v" >> config.env; export "$1=$v"; }
UVV=0.13.9 KSHA=a12d0410e89841e6f3c09958ae5c072f90ae1d49 SSHA=b1956da84bf4a6ccc88f2440078024f1c4bfec7d OSHA=b96fef95e6a746495f92443fac7c688f90fc57fc
RUN=${1:-}
OS=$(uname -s)
if [ ! -f config.env ]; then
  echo "first run: answer or press enter for the default"
  TORCH_BACKEND=$(ask "torch build: rocm7.2 (AMD), cpu, cu128 (NVIDIA), or auto" "$([ "$OS" = Darwin ] && echo cpu || echo rocm7.2)")
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
  { kv ENGINE "$ENGINE"; kv PARAKEET_MODEL "$PARAKEET_MODEL"; kv WHISPER_MODEL "$WHISPER_MODEL"; kv ANCHOR_N 200; kv ANCHOR_EVAL 100; kv TORCH_BACKEND "$TORCH_BACKEND"; kv LANGS "$LANGS"
    kv FLEURS_CFGS "$FLEURS_CFGS"; kv REPLAY_CFGS "$REPLAY_CFGS"; kv REPLAY_N "$REPLAY_N"; kv N_PROMPTS "$N_PROMPTS"; kv MIX_PUBLIC "$MIX_PUBLIC"; kv ACFT_PUBLIC "$ACFT_PUBLIC"; kv DOWNLOAD_DIR "$DOWNLOAD_DIR"; kv OUT_DIR "$OUT_DIR"
    kv HOLDOUT 0.15; kv PORT 8443; kv FT_EPOCHS 3; kv ACFT_EPOCHS 8; kv FORGET_TOL 0.10; } > config.env
  echo "saved to config.env (edit or delete to redo)"
fi
set -a; source config.env; ENGINE=${ENGINE:-whisper}; WHISPER_MODEL=${WHISPER_MODEL:-base}; set +a
[ -n "$RUN" ] || RUN=$(ask "run which workflows: stt | tts | both" "${RUN_LAST:-both}")
case $RUN in stt|tts|both) ;; *) echo "unknown workflow '$RUN': use stt, tts or both"; exit 1 ;; esac
{ grep -v '^RUN_LAST=' config.env || true; echo "RUN_LAST=$RUN"; } > config.env.tmp && mv config.env.tmp config.env
[ "$RUN" = both ] && RUN=stt,tts; export RUN
if [[ $RUN == *tts* ]]; then
  need TTS_LANG "$([[ ",$LANGS," == *,de,* ]] && echo de || echo en)" "TTS voice language: de (German) | en (English)"
  case $TTS_LANG in de) TL=German B=kikiri-tts/kikiri-german-base-51speakers-synthetic/kikiri_german_base_51spk_ep4.pth ;; en) TL=English B=hexgrad/Kokoro-82M/kokoro-v1_0.pth ;; *) echo "TTS_LANG=$TTS_LANG is not supported: use de or en"; exit 1 ;; esac
  [[ ",$LANGS," == *,$TTS_LANG,* ]] || { echo "TTS_LANG=$TTS_LANG: add $TTS_LANG to LANGS in config.env (and its FLEURS config to FLEURS_CFGS)"; exit 1; }
  need TTS_CLIPS 600 "$TL sentences to record for TTS in total (~5.5 s each; the published kikiri voices used 455-627)"
  need TTS_VOICE myvoice "voice name for the TTS output files"
  need TTS_BASE "$B" "TTS base weights: kikiri-tts/kikiri-german-base-51speakers-synthetic/kikiri_german_base_51spk_ep4.pth (German) | hexgrad/Kokoro-82M/kokoro-v1_0.pth (English) | a local .pth"
  need TTS_GATE_MODEL openai/whisper-large-v3-turbo; need TTS_GATE_WER 0.15; need TTS_EPOCHS_1 10; need TTS_EPOCHS_2 10; need TTS_BATCH 4
fi
T="$PWD/.tools"
export PYTHONPATH="$PWD/lib" HF_HOME="$DOWNLOAD_DIR/hf" PATH="$T/bin:$PATH" UV_TOOL_DIR="$T/uv" UV_TOOL_BIN_DIR="$T/bin"
mkdir -p data/wav models "$OUT_DIR" "$DOWNLOAD_DIR" "$T/bin"
have uv || {
  A=$(uname -m); [ "$A" = arm64 ] && A=aarch64; [ "$OS" = Darwin ] && A=$A-apple-darwin || A=$A-unknown-linux-gnu
  case $A in
    x86_64-unknown-linux-gnu) s=a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5 ;;
    aarch64-unknown-linux-gnu) s=0797542018e541a9288855df555c8077eed06f7a47d206362030480e8ecc9f3d ;;
    x86_64-apple-darwin) s=617a3211261b0fc739562f0071aa34e9739797c510208ede8e2977c30903d041 ;;
    aarch64-apple-darwin) s=5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca ;;
    *) echo "uv not found and no pinned build for $A: install uv and rerun"; exit 1 ;;
  esac
  echo "uv not found, installing uv $UVV into $T/bin"; python3 lib/fetch.py "https://github.com/astral-sh/uv/releases/download/$UVV/uv-$A.tar.gz" "$T/bin" "$s"
}
venv() { export VIRTUAL_ENV="$PWD/$1"; [ -d "$1" ] || uv venv --python "$2" "$1"; if [ ! -f "$1/.installed" ]; then uv pip install --torch-backend "$TORCH_BACKEND" -r "$3"; touch "$1/.installed"; fi; }
if [[ $RUN == *tts* ]]; then
  KIKIRI="$DOWNLOAD_DIR/third_party/kikiri-tts-${KSHA:0:7}"; export KIKIRI
  if [ ! -f "$KIKIRI/.ok" ]; then
    rm -rf "$KIKIRI"; G=https://github.com/semidark
    python3 lib/fetch.py "$G/kikiri-tts/archive/$KSHA.tar.gz" "$KIKIRI"; python3 lib/fetch.py "$G/StyleTTS2/archive/$SSHA.tar.gz" "$KIKIRI/StyleTTS2"; python3 lib/fetch.py "$G/kokoro/archive/$OSHA.tar.gz" "$KIKIRI/kokoro"
    touch "$KIKIRI/.ok"
  fi
  venv .venv-tts 3.12 tts/requirements.txt; PYT=.venv-tts/bin/python
  export PYTHONPATH="$PYTHONPATH:$KIKIRI/kokoro:$KIKIRI/StyleTTS2"
  EL=$($PYT -c 'import espeakng_loader, os; print(os.path.dirname(espeakng_loader.__file__))')
  if [ "$OS" = Linux ] && have espeak-ng && [ ! -L "$EL/libespeak-ng.so" ]; then
    LDC=$(command -v ldconfig || echo /sbin/ldconfig); L=$($LDC -p | grep -o '/[^ ]*libespeak-ng\.so\.1$' | head -1); ED=$(espeak-ng --version | sed -n 's/.*Data at: *//p')
    if [ -n "$L" ] && [ -d "$ED" ]; then echo "using the system espeak-ng ($L)"; rm -rf "$EL/espeak-ng-data" "$EL/libespeak-ng.so"; ln -s "$L" "$EL/libespeak-ng.so"; ln -s "$ED" "$EL/espeak-ng-data"; fi
  fi
  $PYT tts/tts_lang.py || { echo "$TL phonemizer failed: install espeak-ng with your package manager and rerun"; exit 1; }
fi
if [[ $RUN == *stt* ]]; then venv ".venv-$ENGINE" "$([ "$ENGINE" = parakeet ] && echo 3.12 || echo 3.13)" "stt/requirements-$ENGINE.txt"; PY=.venv-$ENGINE/bin/python; PYD=$PY; else PYD=$PYT; fi
VIRTUAL_ENV="$PWD/$(dirname "$(dirname "$PYD")")"; export VIRTUAL_ENV
if [ "$TORCH_BACKEND" != cpu ] && [ "${DEVICE:-}" != cpu ] && ! grep -q '^GPU=' config.env; then
  l=$($PYD -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"); GPU=
  if [ "$(printf '%s\n' "$l" | grep -c .)" -gt 1 ]; then echo "$l"; GPU=$(ask "several GPUs found: index to train on (saved as GPU= in config.env)" 0); fi
  kv GPU "$GPU" >> config.env
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
  echo "  ls -l /dev/kfd /dev/dri (you need read/write: add yourself to the video and render groups, re-login; under WSL2 the device is /dev/dxg);"
  echo "  for RX 7700/7800/7600 export HSA_OVERRIDE_GFX_VERSION=11.0.0 and rerun."
  read -r -p "continue on CPU anyway? [y/N]: " c; [ "$c" = y ] || exit 1
fi
hf_get() { local r h=hf; r=$(awk -v k="$1" '$1 == k {print $2}' lib/hf_revisions.txt); have hf || h="$VIRTUAL_ENV/bin/hf"; "$h" download "$1" ${r:+--revision "$r"} >/dev/null; }
if [[ $RUN == *stt* ]]; then if [ "$ENGINE" = parakeet ]; then hf_get "$PARAKEET_MODEL"; else hf_get "openai/whisper-$WHISPER_MODEL"; fi; fi
[ -f data/prompts.txt ] || $PYD record/stage0_data.py
TARGET=$N_PROMPTS
if [[ $RUN == *tts* ]]; then $PYT tts/tts_prompts.py; TARGET=$(wc -l < data/prompts.txt); fi
n=$( [ -f data/manifest.jsonl ] && wc -l < data/manifest.jsonl || echo 0 )
if [ "$TARGET" -gt 0 ] && [ "$n" -lt "$TARGET" ]; then
  echo "recorded $n/$TARGET — starting recorder"
  N_PROMPTS=$TARGET $PYD record/stage1_record.py
fi
DONE=
if [[ $RUN == *tts* ]]; then
  TTS_OUT="$OUT_DIR/kokoro-$TTS_LANG-$TTS_VOICE.pth"
  if [ ! -f "$TTS_OUT" ]; then $PYT tts/tts_prep.py; $PYT tts/tts_train.py; fi
  DONE="$TTS_OUT $OUT_DIR/$TTS_VOICE.pt"
fi
if [[ $RUN != *stt* ]]; then echo "done: $DONE"; exit 0; fi
if [ "$ENGINE" = parakeet ]; then
  [ -f models/parakeet/ft.nemo ] || $PY stt/stage2_parakeet.py
  OUT_GLOB="$OUT_DIR/$(basename "$PARAKEET_MODEL")-*-ft-*.gguf"
else
  [ -f models/ft/config.json ] || $PY stt/stage2_whisper.py
  [ -f models/acft/config.json ] || $PY stt/stage3_whisper_acft.py
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
  if [ "$ENGINE" = parakeet ]; then PY=$PY bash stt/stage4_parakeet.sh; else PY=$PY bash stt/stage4_whisper.sh; fi
fi
echo "done: $(compgen -G "$OUT_GLOB") $DONE"
