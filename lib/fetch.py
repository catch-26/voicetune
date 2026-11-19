import hashlib, io, os, shutil, sys, tarfile, urllib.request
url, dest = sys.argv[1], sys.argv[2]
if not url.startswith("https://"): sys.exit(f"refusing to download over plain http: {url}")
tmp = dest + ".part"; shutil.rmtree(tmp, ignore_errors=True)
data = urllib.request.urlopen(url, timeout=60).read()
if len(sys.argv) > 3 and hashlib.sha256(data).hexdigest() != sys.argv[3]: sys.exit(f"sha256 mismatch for {url}: got {hashlib.sha256(data).hexdigest()}, expected {sys.argv[3]}")
tar = tarfile.open(fileobj=io.BytesIO(data))
if hasattr(tarfile, "data_filter"): tar.extractall(tmp, filter="data")
else:
    bad = [m.name for m in tar.getmembers() if os.path.isabs(m.name) or ".." in m.name.split("/") or m.issym() or m.islnk()]
    if bad: sys.exit(f"refusing to unpack {url}: unsafe entries {bad[:5]}")
    tar.extractall(tmp)
top, = os.listdir(tmp); os.makedirs(dest, exist_ok=True)
for n in os.listdir(os.path.join(tmp, top)): shutil.move(os.path.join(tmp, top, n), os.path.join(dest, n))
shutil.rmtree(tmp)
