import json, os, random, re
from collections import Counter
from common import load
from tts_lang import LANG, NAME, g2p, pool
target = int(os.environ["TTS_CLIPS"])
man = load("data/manifest.jsonl")
old = [r for r in man if r["lang"] == LANG and not os.path.exists(r.get("raw", ""))]
if old and input(f"{len(old)} {NAME} recordings have no full-rate original (16 kHz only, unusable for TTS). re-record them now? [y/N]: ") == "y":
    ids = {r["id"] for r in old}
    with open("data/manifest.jsonl", "w", encoding="utf8") as f:
        for r in man:
            if r["id"] not in ids: f.write(json.dumps(r, ensure_ascii=False) + "\n")
have = [l.split("\t", 1)[1] for l in open("data/prompts.txt", encoding="utf8").read().splitlines() if l.startswith(LANG + "\t")]
if len(have) >= target: raise SystemExit(0)
ok = lambda s: 4 <= len(s.split()) <= 20 and not re.search(r"\d", s)
sents = sorted({s.strip() for s in pool() if ok(s.strip())} - set(have)); random.Random(2).shuffle(sents)
ph = g2p()
di = lambda s: {a + b for w in re.sub("[ˈˌ]", "", ph(s)).split() for a, b in zip(w, w[1:])}
covered = set().union(*map(di, have)) if have else set()
cand = [(s, di(s)) for s in sents[:6000]]; add = []
freq = Counter(d for _, ds in cand for d in ds)
cand = [(s, {d for d in ds if freq[d] >= 20}) for s, ds in cand]
while len(have) + len(add) < target and cand:
    i = max(range(len(cand)), key=lambda k: len(cand[k][1] - covered))
    s, d = cand.pop(i); covered |= d; add.append(s)
with open("data/prompts.txt", "a", encoding="utf8") as f: f.write("".join(f"{LANG}\t{s}\n" for s in add))
print(f"added {len(add)} {NAME} prompts for TTS (picked to cover common {NAME} phoneme pairs, {len(covered)} covered); {NAME} total {len(have) + len(add)}")
