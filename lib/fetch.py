import hashlib, io, os, shutil, sys, tarfile, urllib.request


def tree_sha256(root):
    h = hashlib.sha256()
    for d, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(d, n); rel = os.path.relpath(p, root).replace(os.sep, "/")
            body = ("L" + os.readlink(p)).encode() if os.path.islink(p) else open(p, "rb").read()
            h.update(f"{rel}\0{hashlib.sha256(body).hexdigest()}\n".encode())
    return h.hexdigest()


def fetch(url, dest, want=None):
    if not url.startswith("https://"): sys.exit(f"refusing to download over plain http: {url}")
    tmp = dest + ".part"; shutil.rmtree(tmp, ignore_errors=True)
    data = urllib.request.urlopen(url, timeout=60).read()
    if want and not want.startswith("tree:") and hashlib.sha256(data).hexdigest() != want:
        sys.exit(f"sha256 mismatch for {url}: got {hashlib.sha256(data).hexdigest()}, expected {want}")
    tar = tarfile.open(fileobj=io.BytesIO(data))
    if hasattr(tarfile, "data_filter"): tar.extractall(tmp, filter="data")
    else:
        bad = [m.name for m in tar.getmembers() if os.path.isabs(m.name) or ".." in m.name.split("/") or m.issym() or m.islnk()]
        if bad: sys.exit(f"refusing to unpack {url}: unsafe entries {bad[:5]}")
        tar.extractall(tmp)
    top, = os.listdir(tmp); src = os.path.join(tmp, top)
    if want and want.startswith("tree:") and tree_sha256(src) != want[5:]:
        got = tree_sha256(src); shutil.rmtree(tmp); sys.exit(f"content mismatch for {url}: got tree:{got}, expected {want}")
    os.makedirs(dest, exist_ok=True)
    for n in os.listdir(src): shutil.move(os.path.join(src, n), os.path.join(dest, n))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
