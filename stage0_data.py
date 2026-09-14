import csv, json, os, random, re, subprocess
from datasets import load_dataset, Audio
from common import LANGS, FLEURS_CFGS, REPLAY_CFGS, lang_of
n = int(os.environ["N_PROMPTS"]); dl = os.environ["DOWNLOAD_DIR"]; replay_n = int(os.environ["REPLAY_N"])

def to_wav(ex, wav):
    if not os.path.exists(wav):
        open(f"{dl}/tmp_audio", "wb").write(ex["audio"]["bytes"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{dl}/tmp_audio", "-ac", "1", "-ar", "16000", wav], check=True)

def fetch(cfg, out, limit=None):
    if os.path.exists(out): return
    wavdir = f"{dl}/fleurs_wav/{cfg}"; os.makedirs(wavdir, exist_ok=True)
    ds = load_dataset("google/fleurs", cfg, split="train", revision="refs/convert/parquet", streaming=limit is not None)
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit: ds = ds.take(limit)
    with open(out, "w", encoding="utf8") as f:
        for i, ex in enumerate(ds):
            wav = f"{wavdir}/{i:05d}.wav"; to_wav(ex, wav)
            f.write(json.dumps({"id": i, "lang": lang_of(cfg), "text": ex["raw_transcription"], "wav": wav}, ensure_ascii=False) + "\n")
    print(f"{cfg}: {i + 1} clips -> {out}")

for cfg in FLEURS_CFGS: fetch(cfg, f"data/fleurs.{cfg}.jsonl")
for cfg in REPLAY_CFGS: fetch(cfg, f"data/replay.{cfg}.jsonl", replay_n)

ok = lambda s: 4 <= len(s.split()) <= 20 and not re.search(r"\d", s)
per_lang = {}
for lang, cfg in zip(LANGS, FLEURS_CFGS):
    if os.path.exists(f"data/sentences_raw.{lang}.tsv"):
        lines = [r["sentence"] for r in csv.DictReader(open(f"data/sentences_raw.{lang}.tsv", encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE)]
    elif os.path.exists(f"data/sentences_raw.{lang}.txt"):
        lines = open(f"data/sentences_raw.{lang}.txt", encoding="utf8").read().splitlines()
    else:
        lines = [json.loads(l)["text"] for l in open(f"data/fleurs.{cfg}.jsonl", encoding="utf8")]
    lines = sorted({s.strip() for s in lines if ok(s.strip())}); random.Random(0).shuffle(lines)
    vocab = [v.strip() for v in open(f"data/vocab.{lang}.txt", encoding="utf8").read().splitlines()] if os.path.exists(f"data/vocab.{lang}.txt") else []
    vocab = [v for v in vocab if v]; quota = n // len(LANGS)
    per_lang[lang] = vocab + lines[: max(0, quota - len(vocab))]; random.Random(1).shuffle(per_lang[lang])
out = [f"{lang}\t{per_lang[lang][i]}" for i in range(max(map(len, per_lang.values()))) for lang in LANGS if i < len(per_lang[lang])]
open("data/prompts.txt", "w", encoding="utf8").write("\n".join(out) + "\n")
print(f"{len(out)} prompts: " + ", ".join(f"{k} {len(v)}" for k, v in per_lang.items()))
