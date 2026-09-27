import json, os, random, re, urllib.request
from collections import Counter
from misaki import de
from common import load
CV = "https://raw.githubusercontent.com/common-voice/common-voice/d910b8578b57d74597c5972f217d72abb696c55a/server/data/de/sentence-collector.txt"
target = int(os.environ["TTS_CLIPS"]); pool_f = os.environ["DOWNLOAD_DIR"] + "/cv_de_sentences.txt"
man = load("data/manifest.jsonl")
old = [r for r in man if r["lang"] == "de" and not os.path.exists(r.get("raw", ""))]
if old and input(f"{len(old)} German recordings have no full-rate original (16 kHz only, unusable for TTS). re-record them now? [y/N]: ") == "y":
    ids = {r["id"] for r in old}
    with open("data/manifest.jsonl", "w", encoding="utf8") as f:
        for r in man:
            if r["id"] not in ids: f.write(json.dumps(r, ensure_ascii=False) + "\n")
have = [l.split("\t", 1)[1] for l in open("data/prompts.txt", encoding="utf8").read().splitlines() if l.startswith("de\t")]
if len(have) >= target: raise SystemExit(0)
if not os.path.exists(pool_f): urllib.request.urlretrieve(CV, pool_f)
lines = open(pool_f, encoding="utf8").read().splitlines()
if os.path.exists("data/sentences_raw.de.txt"): lines += open("data/sentences_raw.de.txt", encoding="utf8").read().splitlines()
ok = lambda s: 4 <= len(s.split()) <= 20 and not re.search(r"\d", s)
pool = sorted({s.strip() for s in lines if ok(s.strip())} - set(have)); random.Random(2).shuffle(pool)
g2p = de.DEG2P()
di = lambda s: {a + b for w in re.sub("[ˈˌ]", "", g2p(s)[0]).split() for a, b in zip(w, w[1:])}
covered = set().union(*map(di, have)) if have else set()
cand = [(s, di(s)) for s in pool[:6000]]; add = []
freq = Counter(d for _, ds in cand for d in ds)
cand = [(s, {d for d in ds if freq[d] >= 20}) for s, ds in cand]
while len(have) + len(add) < target and cand:
    i = max(range(len(cand)), key=lambda k: len(cand[k][1] - covered))
    s, d = cand.pop(i); covered |= d; add.append(s)
with open("data/prompts.txt", "a", encoding="utf8") as f: f.write("".join(f"de\t{s}\n" for s in add))
print(f"added {len(add)} German prompts for TTS (picked to cover common German phoneme pairs, {len(covered)} covered); German total {len(have) + len(add)}")
