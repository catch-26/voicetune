#!/usr/bin/env bash
set -euo pipefail
WCSHA=ca19af2449df7a7f4d482fae9d809b9d0817b636 OWSHA=86098128c0b4f24f0e2aa2994de830614b474227
TP="$DOWNLOAD_DIR/third_party"; mkdir -p "$TP" "$OUT_DIR"
get() {
  [ -f "$3/.ok" ] && return; rm -rf "$3"
  if command -v git >/dev/null; then git init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD; else $PY lib/fetch.py "$1/archive/$2.tar.gz" "$3"; fi
  touch "$3/.ok"
}
WC="$TP/whisper.cpp-${WCSHA:0:7}" OW="$TP/whisper-${OWSHA:0:7}"
get https://github.com/ggml-org/whisper.cpp "$WCSHA" "$WC"
get https://github.com/openai/whisper "$OWSHA" "$OW"
$PY "$WC/models/convert-h5-to-ggml.py" models/acft "$OW" "$OUT_DIR"
source stt/ggml_build.sh
B=$(build "$WC" GGML "" whisper-quantize whisper-cli)
name="$OUT_DIR/whisper-$WHISPER_MODEL-$(echo "$LANGS" | tr , -)-acft-q8_0.bin"
"$B/whisper-quantize" "$OUT_DIR/ggml-model.bin" "$name" q8_0
wav=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['wav'])")
ctx=$($PY -c "from common import split, fleurs, audio; a,sr=audio((split()[1] or fleurs()[1])[0]); print(round(1500/30*len(a)/sr))")
lang=$($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['lang'])")
echo "sanity check on $wav with audio_ctx=$ctx:"
"$B/whisper-cli" -m "$name" -l "$lang" -ac "$ctx" -f "$wav"
echo "expected: $($PY -c "from common import split, fleurs; print((split()[1] or fleurs()[1])[0]['text'])")"
