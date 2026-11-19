import csv, json, os, random, re, subprocess
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.errors import RevisionNotFoundError
from hf_pins import rev as pinned
from common import LANGS, FLEURS_CFGS, REPLAY_CFGS, lang_of
n = int(os.environ["N_PROMPTS"]); dl = os.environ["DOWNLOAD_DIR"]; replay_n = int(os.environ["REPLAY_N"])

def to_wav(b, wav):
    if not os.path.exists(wav):
        open(f"{dl}/tmp_audio", "wb").write(b)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{dl}/tmp_audio", "-ac", "1", "-ar", "16000", wav], check=True)

def fetch(cfg, out, limit=None):
    if os.path.exists(out): return
    files, allf = [], []
    for rev in [pinned("datasets/google/fleurs@refs/convert/parquet"), pinned("datasets/google/fleurs@main")]:
        try: allf = list_repo_files("google/fleurs", repo_type="dataset", revision=rev)
        except RevisionNotFoundError: continue
        files = sorted(f for f in allf if f.endswith(".parquet") and f"{cfg}/" in f and "train" in f)
        if files: break
    if not files: raise SystemExit(f"no train parquet for {cfg}. files mentioning it: {[f for f in allf if cfg in f][:20]}")
    wavdir = f"{dl}/fleurs_wav/{cfg}"; os.makedirs(wavdir, exist_ok=True); i = 0
    with open(out, "w", encoding="utf8") as f:
        for fn in files:
            t = pq.read_table(hf_hub_download("google/fleurs", fn, repo_type="dataset", revision=rev))
            col = "raw_transcription" if "raw_transcription" in t.column_names else "transcription"
            for a, txt in zip(t.column("audio").to_pylist(), t.column(col).to_pylist()):
                if limit and i >= limit: break
                wav = f"{wavdir}/{i:05d}.wav"; to_wav(a["bytes"], wav)
                f.write(json.dumps({"id": i, "lang": lang_of(cfg), "text": txt, "wav": wav}, ensure_ascii=False) + "\n"); i += 1
            if limit and i >= limit: break
    print(f"{cfg}: {i} clips from {len(files)} parquet file(s) -> {out}")

main_limit = None if os.environ["MIX_PUBLIC"] == "full" else int(os.environ["ANCHOR_N"]) + int(os.environ["ANCHOR_EVAL"])
for cfg in FLEURS_CFGS: fetch(cfg, f"data/fleurs.{cfg}.jsonl", main_limit)
for cfg in REPLAY_CFGS: fetch(cfg, f"data/replay.{cfg}.jsonl", replay_n)

ok = lambda s: 4 <= len(s.split()) <= 20 and not re.search(r"\d", s)
per_lang = {}
if n == 0: open("data/prompts.txt", "w").close(); raise SystemExit("no recordings requested")
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
