#!/usr/bin/env bash
set -euo pipefail
TP="$DOWNLOAD_DIR/third_party"; mkdir -p "$TP" "$OUT_DIR"
get() { [ -d "$2" ] && return; if command -v git >/dev/null; then git clone --depth 1 "$1" "$2"; else $PY fetch.py "$1/archive/HEAD.tar.gz" "$2"; fi; }
get https://github.com/ggml-org/whisper.cpp "$TP/whisper.cpp"
get https://github.com/openai/whisper "$TP/whisper"
$PY "$TP/whisper.cpp/models/convert-h5-to-ggml.py" models/acft "$TP/whisper" "$OUT_DIR"
source ./ggml_build.sh
B=$(build "$TP/whisper.cpp" GGML "" whisper-quantize whisper-cli)
name="$OUT_DIR/whisper-$WHISPER_MODEL-$(echo "$LANGS" | tr , -)-acft-q8_0.bin"
"$B/whisper-quantize" "$OUT_DIR/ggml-model.bin" "$name" q8_0
wav=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['wav'])")
ctx=$($PY -c "from common import split, fleurs, audio; a,sr=audio((split()[1] or fleurs()[1])[0]); print(round(1500/30*len(a)/sr))")
lang=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['lang'])")
echo "sanity check on $wav with audio_ctx=$ctx:"
"$B/whisper-cli" -m "$name" -l "$lang" -ac "$ctx" -f "$wav"
echo "expected: $($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['text'])")"
