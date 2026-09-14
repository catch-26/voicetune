#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }
if [ ! -f config.env ]; then
  echo "first run: answer or press enter for the default"
  TORCH_BACKEND=$(ask "torch build: rocm7.2 (AMD), cpu, cu128 (NVIDIA), or auto" rocm7.2)
  LANGS=$(ask "languages you will record, ISO codes, comma separated" "en,de")
  FLEURS_CFGS=$(ask "matching FLEURS configs (huggingface.co/datasets/google/fleurs)" "en_us,de_de")
  REPLAY_CFGS=$(ask "other-language FLEURS configs to protect from forgetting (empty = none)" "es_419,it_it,ru_ru")
  REPLAY_N=$(ask "clips per protected language (half train, half check)" 150)
  N_PROMPTS=$(ask "sentences to record in total (~5.5 s each; 400 = 35 min, 650 = 1 h)" 400)
  MIX_PUBLIC=$(ask "FLEURS audio of your languages in the finetune: none | anchor (~200 clips/lang, keeps other speakers good) | full (10 h/lang, retune the base model, no recordings needed)" anchor)
  ACFT_PUBLIC=$(ask "use FLEURS for the ACFT step (recommended)" y)
  DOWNLOAD_DIR=$(ask "download folder" "$PWD/downloads")
  OUT_DIR=$(ask "output folder for the FUTO model" "$PWD/out")
  printf 'ANCHOR_N=200\nANCHOR_EVAL=100\nTORCH_BACKEND=%s\nLANGS=%s\nFLEURS_CFGS=%s\nREPLAY_CFGS=%s\nREPLAY_N=%s\nN_PROMPTS=%s\nMIX_PUBLIC=%s\nACFT_PUBLIC=%s\nDOWNLOAD_DIR=%s\nOUT_DIR=%s\nHOLDOUT=0.15\nPORT=8443\nFT_EPOCHS=3\nACFT_EPOCHS=8\nFORGET_TOL=0.10\n' \
    "$TORCH_BACKEND" "$LANGS" "$FLEURS_CFGS" "$REPLAY_CFGS" "$REPLAY_N" "$N_PROMPTS" "$MIX_PUBLIC" "$ACFT_PUBLIC" "$DOWNLOAD_DIR" "$OUT_DIR" > config.env
  echo "saved to config.env (edit or delete to redo)"
fi
set -a; source config.env; set +a
export HF_HOME="$DOWNLOAD_DIR/hf"
mkdir -p data/wav models "$OUT_DIR" "$DOWNLOAD_DIR"
command -v uv >/dev/null || { echo "uv not found: install it from astral.sh/uv"; exit 1; }
[ -d .venv ] || uv venv --python 3.13
if [ ! -f .venv/.installed ]; then
  if [ "$TORCH_BACKEND" = auto ]; then uv pip install --torch-backend=auto -r requirements.txt
  else uv pip install --index "pytorch=https://download.pytorch.org/whl/$TORCH_BACKEND" -r requirements.txt; fi
  touch .venv/.installed
fi
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, using CPU (slow but works)')"
if [ "$TORCH_BACKEND" != cpu ] && ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "GPU not visible to torch. checks: torch version above must end in +$TORCH_BACKEND (else: rm -rf .venv and rerun);"
  echo "  ls -l /dev/kfd /dev/dri (you need read/write: add yourself to the video and render groups, re-login);"
  echo "  for RX 7700/7800/7600 export HSA_OVERRIDE_GFX_VERSION=11.0.0 and rerun."
  read -r -p "continue on CPU anyway? [y/N]: " c; [ "$c" = y ] || exit 1
fi
[ -f data/prompts.txt ] || $PY stage0_data.py
n=$( [ -f data/manifest.jsonl ] && wc -l < data/manifest.jsonl || echo 0 )
if [ "$N_PROMPTS" -gt 0 ] && [ "$n" -lt "$N_PROMPTS" ]; then
  echo "recorded $n/$N_PROMPTS — starting recorder"
  $PY stage1_record.py
fi
[ -f models/ft/config.json ] || $PY stage2_finetune.py
[ -f models/acft/config.json ] || $PY stage3_acft.py
ls "$OUT_DIR"/*q8_0.bin >/dev/null 2>&1 || PY=$PY bash stage4_export.sh
echo "done: $(ls "$OUT_DIR"/*q8_0.bin)"
