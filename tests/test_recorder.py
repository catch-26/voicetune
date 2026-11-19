import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = ssl._create_unverified_context()
KEY = "testkey1"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(tmp_path, run, n=2):
    (tmp_path / "data/wav").mkdir(parents=True)
    (tmp_path / "data/prompts.txt").write_text("en\tthe first test sentence\nde\tder zweite Testsatz hier\nen\ta third one\n", encoding="utf8")
    port = free_port()
    env = dict(os.environ, PORT=str(port), N_PROMPTS=str(n), RUN=run, RECORDER_KEY=KEY)
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "record", "stage1_record.py")], cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"https://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/next?k=" + KEY, context=CTX, timeout=2)
            return p, base
        except OSError:
            if p.poll() is not None:
                pytest.fail("recorder exited: " + p.stdout.read())
            time.sleep(0.2)
    p.kill()
    pytest.fail("recorder did not start")


def k(url):
    return url + ("&" if "?" in url else "?") + "k=" + KEY


def get(url):
    return urllib.request.urlopen(k(url), context=CTX, timeout=10).read()


def post(url, data=b""):
    return urllib.request.urlopen(urllib.request.Request(k(url), data=data, method="POST"), context=CTX, timeout=30).read()


@pytest.mark.parametrize("run,mic", [("stt", "{audio:true}"), ("stt,tts", "{audio:{echoCancellation:false")])
def test_page_and_next(tmp_path, run, mic):
    p, base = start(tmp_path, run)
    try:
        page = get(base + "/").decode()
        assert mic in page and "__MIC__" not in page and "__REC__" not in page
        assert json.loads(get(base + "/next")) == {"id": 0, "lang": "en", "text": "the first test sentence", "n": 1, "total": 2}
    finally:
        p.kill()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_save_retake_and_exit(tmp_path):
    clip = tmp_path / "clip.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=1", str(clip)], check=True)
    p, base = start(tmp_path, "stt")
    try:
        post(base + "/save?id=0", clip.read_bytes())
        rows = [json.loads(line) for line in (tmp_path / "data/manifest.jsonl").read_text(encoding="utf8").splitlines()]
        assert rows == [{"id": 0, "lang": "en", "text": "the first test sentence", "wav": "data/wav/00000.wav", "raw": "data/raw/00000.webm"}]
        assert get(base + "/wav?id=0")[:4] == b"RIFF"
        assert json.loads(get(base + "/next"))["id"] == 1
        post(base + "/retake?id=0")
        assert json.loads(get(base + "/next"))["id"] == 0
        post(base + "/save?id=0", clip.read_bytes())
        try:
            post(base + "/save?id=1", clip.read_bytes())
        except OSError:
            pass
        assert p.wait(timeout=20) == 0
        assert len((tmp_path / "data/manifest.jsonl").read_text().splitlines()) == 2
    finally:
        p.kill()


def status(url, data=None, key=True):
    try:
        urllib.request.urlopen(urllib.request.Request(k(url) if key else url, data=data, method="POST" if data is not None else "GET"), context=CTX, timeout=10)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def test_rejects_bad_ids_and_paths(tmp_path):
    p, base = start(tmp_path, "stt")
    try:
        assert status(base + "/save?id=2", b"x") == 400
        assert status(base + "/save?id=-1", b"x") == 400
        assert status(base + "/retake?id=abc", b"") == 400
        assert status(base + "/other?id=0", b"x") == 400
        assert status(base + "/wav?id=0") == 404
        assert status(base + "/wav?id=99") == 404
        assert not (tmp_path / "data/manifest.jsonl").exists()
    finally:
        p.kill()


def test_needs_access_code(tmp_path):
    p, base = start(tmp_path, "stt")
    try:
        assert status(base + "/", key=False) == 403
        assert status(base + "/next?k=wrong", key=False) == 403
        assert status(base + "/save?id=0", b"x", key=False) == 403
        r = urllib.request.urlopen(base + "/?k=" + KEY, context=CTX, timeout=10)
        cookie = r.headers["Set-Cookie"]
        assert cookie.startswith("k=" + KEY) and "SameSite=Strict" in cookie and "HttpOnly" in cookie
        req = urllib.request.Request(base + "/next", headers={"Cookie": "k=" + KEY})
        assert json.loads(urllib.request.urlopen(req, context=CTX, timeout=10).read())["id"] == 0
    finally:
        p.kill()
