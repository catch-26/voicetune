import http.server, json, os, shutil, ssl, subprocess, sys, socket
from urllib.parse import urlparse, parse_qs
PORT = int(os.environ["PORT"]); N = int(os.environ["N_PROMPTS"])
prompts = [l.split("\t", 1) for l in open("data/prompts.txt", encoding="utf8").read().splitlines()]
man = {}
if os.path.exists("data/manifest.jsonl"):
    for l in open("data/manifest.jsonl", encoding="utf8"):
        r = json.loads(l); man[r["id"]] = r
if not os.path.exists("cert.pem") and shutil.which("openssl"):
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "key.pem", "-out", "cert.pem", "-days", "3650", "-subj", "/CN=rec"], check=True, stderr=subprocess.DEVNULL)
elif not os.path.exists("cert.pem"):
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "rec")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=3650)).sign(key, hashes.SHA256())
    open("key.pem", "wb").write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    open("cert.pem", "wb").write(cert.public_bytes(serialization.Encoding.PEM))

PAGE = """<!doctype html><meta name=viewport content="width=device-width">
<body style="font-family:sans-serif;margin:24px;text-align:center">
<div id=setup>
<h3>mic check</h3>
<p>hold the phone as you do when dictating. tap, say one sentence, tap again, then listen.</p>
<button id=tb style="font-size:1.3em;padding:16px 30px">test record</button>
<p><audio id=ta controls></audio></p>
<p style="text-align:left;font-size:0.95em">tips: normal dictation pace, no extra pauses; say punctuation only if you would dictate it; a misread sentence is fine, just tap retake; take a break every 50.</p>
<button id=go style="font-size:1.3em;padding:16px 30px" disabled>sounds fine, start</button>
</div>
<div id=main hidden>
<div id=c></div><p id=t style="font-size:1.6em;min-height:4em"></p>
<button id=b style="font-size:1.5em;padding:24px 40px">record</button>
<p><button id=r>retake last</button> <button id=pl>play last</button></p>
</div>
<script>
let id, mr, chunks, lastId, stream;
const rec = async (onstop) => {stream = stream || await navigator.mediaDevices.getUserMedia({audio:true});
 chunks=[]; mr=new MediaRecorder(stream); mr.ondataavailable=e=>chunks.push(e.data); mr.onstop=onstop; mr.start()};
tb.onclick=async()=>{if(mr&&mr.state=='recording'){mr.stop();tb.textContent='test record';return}
 await rec(()=>{ta.src=URL.createObjectURL(new Blob(chunks));go.disabled=false}); tb.textContent='stop'};
go.onclick=()=>{setup.hidden=true;main.hidden=false;next()};
async function next(){const j=await (await fetch('/next')).json();
 if(j.done){t.textContent='all done, close this page';b.disabled=true;return}
 id=j.id;c.textContent=j.n+' / '+j.total+'   ['+j.lang+']';t.textContent=j.text}
b.onclick=async()=>{if(mr&&mr.state=='recording'){mr.stop();b.textContent='record';b.style.background='';return}
 await rec(async()=>{await fetch('/save?id='+id,{method:'POST',body:new Blob(chunks)});lastId=id;await next()});
 b.textContent='stop';b.style.background='#f66'};
r.onclick=async()=>{if(lastId==null)return;await fetch('/retake?id='+lastId,{method:'POST'});await next()};
pl.onclick=()=>{if(lastId!=null)new Audio('/wav?id='+lastId).play()};
</script>"""

def write_manifest():
    with open("data/manifest.jsonl", "w", encoding="utf8") as f:
        for r in sorted(man.values(), key=lambda r: r["id"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")

class H(http.server.BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", len(b)); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/": return self.send(PAGE, "text/html; charset=utf-8")
        if p == "/next":
            todo = [i for i in range(min(N, len(prompts))) if i not in man]
            if not todo: return self.send(json.dumps({"done": True}))
            i = todo[0]; return self.send(json.dumps({"id": i, "lang": prompts[i][0], "text": prompts[i][1], "n": len(man) + 1, "total": min(N, len(prompts))}))
        if p == "/wav":
            i = int(parse_qs(urlparse(self.path).query)["id"][0]); return self.send(open(f"data/wav/{i:05d}.wav", "rb").read(), "audio/wav")
        self.send_response(404); self.end_headers()
    def do_POST(self):
        u = urlparse(self.path); i = int(parse_qs(u.query)["id"][0])
        if u.path == "/retake":
            man.pop(i, None); write_manifest(); return self.send("{}")
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        src = "data/tmp.webm"; open(src, "wb").write(raw)
        wav = f"data/wav/{i:05d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", wav], check=True)
        man[i] = {"id": i, "lang": prompts[i][0], "text": prompts[i][1], "wav": wav}; write_manifest(); self.send("{}")
        if len(man) >= min(N, len(prompts)):
            print("target reached, exiting recorder"); sys.stdout.flush(); os._exit(0)
    def log_message(self, *a): pass

srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain("cert.pem", "key.pem")
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_.connect(("8.8.8.8", 80)); ip = s_.getsockname()[0]
print(f"open https://{ip}:{PORT}/ on the phone (accept the self-signed cert warning)")
srv.serve_forever()
