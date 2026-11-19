import json, os, re, soundfile as sf, torch
DEV = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
LANGS = os.environ["LANGS"].split(",")
FLEURS_CFGS = os.environ["FLEURS_CFGS"].split(",")
REPLAY_CFGS = [c for c in os.environ["REPLAY_CFGS"].split(",") if c]
HOLDOUT = float(os.environ["HOLDOUT"])
ENGINE = os.environ.get("ENGINE", "whisper")
BASE = os.environ["PARAKEET_MODEL"] if ENGINE == "parakeet" else "openai/whisper-" + os.environ.get("WHISPER_MODEL", "base")
EN_ONLY = BASE.endswith(".en")
def lang_kw(lang):
    return {} if EN_ONLY else {"language": lang, "task": "transcribe"}
def lang_of(cfg):
    c = cfg.split("_")[0]
    return {"cmn": "zh", "fil": "tl", "ast": "es"}.get(c, c)
def load(path):
    return [json.loads(l) for l in open(path, encoding="utf8")] if os.path.exists(path) else []
def split():
    r = load("data/manifest.jsonl")
    hold = lambda x: x["id"] % 100 < HOLDOUT * 100
    return [x for x in r if not hold(x)], [x for x in r if hold(x)]
ANCHOR_EVAL = int(os.environ["ANCHOR_EVAL"])
def fleurs():
    tr, ev = [], []
    for c in FLEURS_CFGS:
        r = load(f"data/fleurs.{c}.jsonl"); tr += r[:-ANCHOR_EVAL]; ev += r[-ANCHOR_EVAL:]
    return tr, ev
def replay():
    tr, ev = [], []
    for c in REPLAY_CFGS:
        r = load(f"data/replay.{c}.jsonl"); h = len(r) // 2
        tr += r[:h]; ev += r[h:]
    return tr, ev
def audio(r):
    return sf.read(r["wav"], dtype="float32")
def norm(s):
    return re.sub(r"[^\w\s]", "", s.lower().replace("ß", "ss")).split()
