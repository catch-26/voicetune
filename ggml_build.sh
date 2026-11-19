build() {
  local src=$1 pre=$2 extra=$3 b d f arch; shift 3
  for b in vulkan hip cpu; do
    [ "${DEVICE:-}" = cpu ] && [ "$b" != cpu ] && continue
    d="$src/build-$b"
    [ -f "$d/.ok" ] && { echo "$d/bin"; return; }
    case $b in
      vulkan) f="-D${pre}_VULKAN=ON" ;;
      hip) command -v rocminfo >/dev/null || continue; arch=$(rocminfo | grep -o 'gfx[0-9a-f]*' | grep -v gfx000 | head -1); [ -n "$arch" ] || continue; f="-D${pre}_HIP=ON -DAMDGPU_TARGETS=$arch" ;;
      cpu) f= ;;
    esac
    echo "building $(basename "$src") for $b" >&2
    if cmake -S "$src" -B "$d" -DCMAKE_BUILD_TYPE=Release $extra $f >&2 && cmake --build "$d" -j --target "$@" >&2; then touch "$d/.ok"; echo "$d/bin"; return; fi
    echo "$b build failed, trying the next backend" >&2
  done
  return 1
}
