#!/usr/bin/env bash
set -euo pipefail
TP="$DOWNLOAD_DIR/third_party"; mkdir -p "$TP" "$OUT_DIR"
source ./ggml_build.sh
get() { [ -d "$2" ] && return; if command -v git >/dev/null; then git clone --depth 1 "$1" "$2"; else $PY fetch.py "$1/archive/HEAD.tar.gz" "$2"; fi; }
get https://github.com/handy-computer/transcribe.cpp "$TP/transcribe.cpp"
slug=$(basename "$PARAKEET_MODEL" .nemo)
conv="$TP/conv/$slug"; mkdir -p "$conv"
$PY "$TP/transcribe.cpp/scripts/convert-parakeet.py" models/parakeet/ft.nemo "$conv/$slug-F32.gguf"
B=$(build "$TP/transcribe.cpp" TRANSCRIBE "-DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF" transcribe-quantize transcribe-cli)
q=Q4_K_M; case $slug in *110m) q=Q8_0 ;; esac
name="$OUT_DIR/$slug-$(echo "$LANGS" | tr , -)-ft-$q.gguf"
"$B/transcribe-quantize" "$conv/$slug-F32.gguf" "$name" --quant "$q"
rm -f "$conv/$slug-F32.gguf"
wav=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['wav'])")
lang=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['lang'])")
echo "sanity check on $wav:"
"$B/transcribe-cli" -m "$name" -l "$lang" "$wav"
echo "expected: $($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['text'])")"
