import io, os, shutil, sys, tarfile, urllib.request
url, dest = sys.argv[1], sys.argv[2]
tmp = dest + ".part"; shutil.rmtree(tmp, ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall(tmp)
top, = os.listdir(tmp); os.makedirs(dest, exist_ok=True)
for n in os.listdir(os.path.join(tmp, top)): shutil.move(os.path.join(tmp, top, n), os.path.join(dest, n))
shutil.rmtree(tmp)
