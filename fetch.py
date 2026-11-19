import io, os, shutil, sys, tarfile, urllib.request
url, dest = sys.argv[1], sys.argv[2]
if not url.startswith("https://"): sys.exit(f"refusing to download over plain http: {url}")
tmp = dest + ".part"; shutil.rmtree(tmp, ignore_errors=True)
tar = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
if hasattr(tarfile, "data_filter"): tar.extractall(tmp, filter="data")
else:
    bad = [m.name for m in tar.getmembers() if os.path.isabs(m.name) or ".." in m.name.split("/") or m.issym() or m.islnk()]
    if bad: sys.exit(f"refusing to unpack {url}: unsafe entries {bad[:5]}")
    tar.extractall(tmp)
top, = os.listdir(tmp); os.makedirs(dest, exist_ok=True)
for n in os.listdir(os.path.join(tmp, top)): shutil.move(os.path.join(tmp, top, n), os.path.join(dest, n))
shutil.rmtree(tmp)
