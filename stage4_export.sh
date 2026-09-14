#!/usr/bin/env bash
set -euo pipefail
TP="$DOWNLOAD_DIR/third_party"; mkdir -p "$TP" "$OUT_DIR"
[ -d "$TP/whisper.cpp" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$TP/whisper.cpp"
[ -d "$TP/whisper" ] || git clone --depth 1 https://github.com/openai/whisper "$TP/whisper"
$PY "$TP/whisper.cpp/models/convert-h5-to-ggml.py" models/acft "$TP/whisper" "$OUT_DIR"
[ -x "$TP/whisper.cpp/build/bin/quantize" ] || { cmake -S "$TP/whisper.cpp" -B "$TP/whisper.cpp/build"; cmake --build "$TP/whisper.cpp/build" -j --config Release; }
name="$OUT_DIR/whisper-base-$(echo "$LANGS" | tr , -)-acft-q8_0.bin"
"$TP/whisper.cpp/build/bin/quantize" "$OUT_DIR/ggml-model.bin" "$name" q8_0
wav=$($PY -c "from common import split; print(split()[1][0]['wav'])")
ctx=$($PY -c "from common import split, audio; a,sr=audio(split()[1][0]); print(round(1500/30*len(a)/sr))")
lang=$($PY -c "from common import split; print(split()[1][0]['lang'])")
echo "sanity check on $wav with audio_ctx=$ctx:"
"$TP/whisper.cpp/build/bin/whisper-cli" -m "$name" -l "$lang" -ac "$ctx" -f "$wav"
echo "expected: $($PY -c "from common import split; print(split()[1][0]['text'])")"
