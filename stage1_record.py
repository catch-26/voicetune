import http.server, json, os, shutil, ssl, subprocess, sys, socket
from urllib.parse import urlparse, parse_qs
PORT = int(os.environ["PORT"]); N = int(os.environ["N_PROMPTS"])
TTS = "tts" in os.environ.get("RUN", "")
os.makedirs("data/raw", exist_ok=True)
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

PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "recorder.html"), encoding="utf8").read()

MIC = "{echoCancellation:false,noiseSuppression:false,autoGainControl:false}" if TTS else "true"
REC = "{audioBitsPerSecond:128000}" if TTS else "{}"

def write_manifest():
    with open("data/manifest.jsonl", "w", encoding="utf8") as f:
        for r in sorted(man.values(), key=lambda r: r["id"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")

class H(http.server.BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", len(b)); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/": return self.send(PAGE.replace("__MIC__", MIC).replace("__REC__", REC), "text/html; charset=utf-8")
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
        src = f"data/raw/{i:05d}.webm"; open(src, "wb").write(raw)
        wav = f"data/wav/{i:05d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", wav], check=True)
        man[i] = {"id": i, "lang": prompts[i][0], "text": prompts[i][1], "wav": wav, "raw": src}; write_manifest(); self.send("{}")
        if len(man) >= min(N, len(prompts)):
            print("target reached, exiting recorder"); sys.stdout.flush(); os._exit(0)
    def log_message(self, *a): pass

srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain("cert.pem", "key.pem")
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_.connect(("8.8.8.8", 80)); ip = s_.getsockname()[0]
print(f"open https://{ip}:{PORT}/ on the phone (accept the self-signed cert warning)")
srv.serve_forever()
