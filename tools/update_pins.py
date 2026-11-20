"""Update or verify the pins that Dependabot cannot see.

Without arguments: move each pin to the newest upstream version that is at
least GRACE_DAYS old, rewrite the files and print a Markdown summary.
With --verify: download every hashed archive and file and check its hash.
"""
import datetime, hashlib, json, os, re, sys, tempfile, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))
from fetch import fetch, tree_sha256  # noqa: E402

GRACE_DAYS = 7
CUTOFF = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=GRACE_DAYS)
UV_TARGETS = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-apple-darwin", "aarch64-apple-darwin"]
CV_PATHS = {"de": "server/data/de/sentence-collector.txt", "en": "server/data/en/sentence-collector.txt"}
GIT = [
    ("run.sh", "KSHA", "KTREE", "semidark/kikiri-tts"),
    ("run.sh", "SSHA", "STREE", "semidark/StyleTTS2"),
    ("run.sh", "OSHA", "OTREE", "semidark/kokoro"),
    ("stt/stage4_whisper.sh", "WCSHA", "WCTREE", "ggml-org/whisper.cpp"),
    ("stt/stage4_whisper.sh", "OWSHA", "OWTREE", "openai/whisper"),
    ("stt/stage4_parakeet.sh", "TCSHA", "TCTREE", "handy-computer/transcribe.cpp"),
]
URL_REQS = [("semidark/misaki", ["tts/tts_lang.py"]), ("resemble-ai/monotonic_align", [])]


def path(f): return os.path.join(ROOT, f)
def read(f): return open(path(f), encoding="utf8").read()
def write(f, s): open(path(f), "w", encoding="utf8").write(s)
def when(s): return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def get(url, raw=False):
    h = {"User-Agent": "voicetune-pins"}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"): h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    data = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60).read()
    return data if raw else json.loads(data)


def var(f, name): return re.search(rf"\b{name}=([0-9a-f]+)", read(f)).group(1)


def gh_commit(repo, sha): return get(f"https://api.github.com/repos/{repo}/commits/{sha}")


def gh_newest(repo, current, paths=(None,)):
    """Newest default-branch commit at least GRACE_DAYS old, if newer than current."""
    best = None
    for p in paths:
        q = {"until": CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ"), "per_page": 1}
        if p: q["path"] = p
        c = get(f"https://api.github.com/repos/{repo}/commits?" + urllib.parse.urlencode(q))
        if c and (best is None or when(c[0]["commit"]["committer"]["date"]) > when(best["commit"]["committer"]["date"])): best = c[0]
    if not best or best["sha"] == current: return None
    if when(best["commit"]["committer"]["date"]) <= when(gh_commit(repo, current)["commit"]["committer"]["date"]): return None
    return best["sha"], best["commit"]["committer"]["date"][:10]


def archive_tree(repo, sha):
    with tempfile.TemporaryDirectory() as d:
        fetch(f"https://github.com/{repo}/archive/{sha}.tar.gz", os.path.join(d, "x"))
        return tree_sha256(os.path.join(d, "x"))


def cv_url(sha, lang): return f"https://raw.githubusercontent.com/common-voice/common-voice/{sha}/{CV_PATHS[lang]}"


def cv_pins():
    s = read("tts/tts_lang.py")
    return re.search(r"common-voice/common-voice/([0-9a-f]{40})/", s).group(1), dict(re.findall(r'"(\w\w)": "([0-9a-f]{64})"', s))


def hf_newest(key, current):
    repo, _, ref = key.partition("@")
    kind, name = ("datasets", repo[len("datasets/"):]) if repo.startswith("datasets/") else ("models", repo)
    commits = get(f"https://huggingface.co/api/{kind}/{name}/commits/{urllib.parse.quote(ref or 'main', safe='')}")
    ids = [c["id"] for c in commits]
    old = [c for c in commits if when(c["date"]) <= CUTOFF]
    if not old or old[0]["id"] == current: return None
    if current in ids and ids.index(old[0]["id"]) > ids.index(current): return None
    return old[0]["id"], old[0]["date"][:10]


def update():
    notes = []
    for f, sv, tv, repo in GIT:
        cur = var(f, sv); new = gh_newest(repo, cur)
        if not new: continue
        sha, day = new; s = read(f)
        s = s.replace(f"{sv}={cur}", f"{sv}={sha}").replace(f"{tv}={var(f, tv)}", f"{tv}={archive_tree(repo, sha)}")
        write(f, s); notes.append(f"- {repo}: {cur[:7]} -> {sha[:7]} ({day})")
    for repo, short_files in URL_REQS:
        s = read("tts/requirements.txt"); cur = re.search(rf"github.com/{repo}/archive/([0-9a-f]{{40}})", s).group(1); new = gh_newest(repo, cur)
        if not new: continue
        sha, day = new; write("tts/requirements.txt", s.replace(cur, sha))
        for g in short_files: write(g, read(g).replace(cur[:7], sha[:7]))
        notes.append(f"- {repo}: {cur[:7]} -> {sha[:7]} ({day})")
    cur, hashes = cv_pins(); new = gh_newest("common-voice/common-voice", cur, list(CV_PATHS.values()))
    if new:
        sha, day = new; s = read("tts/tts_lang.py").replace(cur, sha)
        for lang, h in hashes.items(): s = s.replace(h, hashlib.sha256(get(cv_url(sha, lang), raw=True)).hexdigest())
        write("tts/tts_lang.py", s); notes.append(f"- Common Voice sentence lists: {cur[:7]} -> {sha[:7]} ({day})")
    s = read("run.sh"); uvv = re.search(r"\bUVV=([0-9.]+)", s).group(1)
    rel = [r for r in get("https://api.github.com/repos/astral-sh/uv/releases?per_page=30") if not r["draft"] and not r["prerelease"] and r["published_at"] and when(r["published_at"]) <= CUTOFF]
    cur_rel = get(f"https://api.github.com/repos/astral-sh/uv/releases/tags/{uvv}")
    if rel:
        r = max(rel, key=lambda r: when(r["published_at"]))
        if r["tag_name"] != uvv and when(r["published_at"]) > when(cur_rel["published_at"]):
            assets = {a["name"]: a["browser_download_url"] for a in r["assets"]}
            for t in UV_TARGETS:
                old = re.search(rf"{t}\) s=([0-9a-f]{{64}})", s).group(1)
                s = s.replace(old, get(assets[f"uv-{t}.tar.gz.sha256"], raw=True).decode().split()[0])
            write("run.sh", s.replace(f"UVV={uvv}", f"UVV={r['tag_name']}")); notes.append(f"- uv: {uvv} -> {r['tag_name']} ({r['published_at'][:10]})")
    lines = read("lib/hf_revisions.txt").splitlines(); out = []
    for line in lines:
        parts = line.split("#")[0].split()
        if len(parts) == 2:
            new = hf_newest(*parts)
            if new:
                line = line.replace(parts[1], new[0]); notes.append(f"- {parts[0]}: {parts[1][:7]} -> {new[0][:7]} ({new[1]})")
        out.append(line)
    write("lib/hf_revisions.txt", "\n".join(out) + "\n")
    print("\n".join(notes))


def verify():
    bad = 0
    for f, sv, tv, repo in GIT:
        sha, want = var(f, sv), var(f, tv); got = archive_tree(repo, sha)
        print(f"{repo}@{sha[:7]}: {'ok' if got == want else 'MISMATCH ' + got}"); bad += got != want
    sha, hashes = cv_pins()
    for lang, want in hashes.items():
        got = hashlib.sha256(get(cv_url(sha, lang), raw=True)).hexdigest()
        print(f"Common Voice {lang}@{sha[:7]}: {'ok' if got == want else 'MISMATCH ' + got}"); bad += got != want
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    verify() if sys.argv[1:] == ["--verify"] else update()
