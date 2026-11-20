#!/usr/bin/env bash
set -euo pipefail
TCSHA=adfc50210a3f026a655a3e19937d90ef2bf0728c
TCTREE=0cae37b0f7a76e4f9baa1422ff006a934e2cb926ca6471eab2fef67ea9c80297
TP="$DOWNLOAD_DIR/third_party"; mkdir -p "$TP" "$OUT_DIR"
source stt/ggml_build.sh
get() {
  [ -f "$3/.ok" ] && return; rm -rf "$3"
  if command -v git >/dev/null; then git init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD; else $PY lib/fetch.py "$1/archive/$2.tar.gz" "$3" "tree:$4"; fi
  touch "$3/.ok"
}
TC="$TP/transcribe.cpp-${TCSHA:0:7}"; get https://github.com/handy-computer/transcribe.cpp "$TCSHA" "$TC" "$TCTREE"
slug=$(basename "$PARAKEET_MODEL" .nemo)
conv="$TP/conv/$slug"; mkdir -p "$conv"
$PY "$TC/scripts/convert-parakeet.py" models/parakeet/ft.nemo "$conv/$slug-F32.gguf"
B=$(build "$TC" TRANSCRIBE "-DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_BUILD_TESTS=OFF" transcribe-quantize transcribe-cli)
q=Q4_K_M; case $slug in *110m) q=Q8_0 ;; esac
name="$OUT_DIR/$slug-$(echo "$LANGS" | tr , -)-ft-$q.gguf"
"$B/transcribe-quantize" "$conv/$slug-F32.gguf" "$name" --quant "$q"
rm -f "$conv/$slug-F32.gguf"
wav=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['wav'])")
lang=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['lang'])")
echo "sanity check on $wav:"
"$B/transcribe-cli" -m "$name" -l "$lang" "$wav"
echo "expected: $($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['text'])")"
