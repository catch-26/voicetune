#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ask() { local v; read -r -p "$1 [$2]: " v; echo "${v:-$2}"; }
if [ ! -f config.env ]; then
  echo "first run: answer or press enter for the default"
  LANGS=$(ask "languages you will record, ISO codes, comma separated" "en,de")
  FLEURS_CFGS=$(ask "matching FLEURS configs (huggingface.co/datasets/google/fleurs)" "en_us,de_de")
  REPLAY_CFGS=$(ask "other-language FLEURS configs to protect from forgetting (empty = none)" "fr_fr,es_419,it_it")
  REPLAY_N=$(ask "clips per protected language (half train, half check)" 150)
  N_PROMPTS=$(ask "sentences to record in total (~5.5 s each; 400 = 35 min, 650 = 1 h)" 400)
  MIX_PUBLIC=$(ask "also train on FLEURS audio of your languages? y = language model, n = your voice only" n)
  ACFT_PUBLIC=$(ask "use FLEURS for the ACFT step (recommended)" y)
  DOWNLOAD_DIR=$(ask "download folder" "$PWD/downloads")
  OUT_DIR=$(ask "output folder for the FUTO model" "$PWD/out")
  printf 'LANGS=%s\nFLEURS_CFGS=%s\nREPLAY_CFGS=%s\nREPLAY_N=%s\nN_PROMPTS=%s\nMIX_PUBLIC=%s\nACFT_PUBLIC=%s\nDOWNLOAD_DIR=%s\nOUT_DIR=%s\nHOLDOUT=0.15\nPORT=8443\nFT_EPOCHS=3\nACFT_EPOCHS=8\nFORGET_TOL=0.10\n' \
    "$LANGS" "$FLEURS_CFGS" "$REPLAY_CFGS" "$REPLAY_N" "$N_PROMPTS" "$MIX_PUBLIC" "$ACFT_PUBLIC" "$DOWNLOAD_DIR" "$OUT_DIR" > config.env
  echo "saved to config.env (edit or delete to redo)"
fi
set -a; source config.env; set +a
export HF_HOME="$DOWNLOAD_DIR/hf"
mkdir -p data/wav models "$OUT_DIR" "$DOWNLOAD_DIR"
command -v uv >/dev/null || { echo "uv not found: install it from astral.sh/uv"; exit 1; }
[ -d .venv ] || uv venv --python 3.13
[ -f .venv/.installed ] || { uv pip install --torch-backend=auto -r requirements.txt && touch .venv/.installed; }
PY=.venv/bin/python
$PY -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, using CPU (slow but works)')"
[ -f data/prompts.txt ] || $PY stage0_data.py
n=$( [ -f data/manifest.jsonl ] && wc -l < data/manifest.jsonl || echo 0 )
if [ "$n" -lt "$N_PROMPTS" ]; then
  echo "recorded $n/$N_PROMPTS — starting recorder"
  $PY stage1_record.py
fi
[ -f models/ft/config.json ] || $PY stage2_finetune.py
[ -f models/acft/config.json ] || $PY stage3_acft.py
ls "$OUT_DIR"/*q8_0.bin >/dev/null 2>&1 || PY=$PY bash stage4_export.sh
echo "done: $(ls "$OUT_DIR"/*q8_0.bin)"
