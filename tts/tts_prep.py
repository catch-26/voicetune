import json, os, subprocess, librosa, jiwer, soundfile as sf, torch
from collections import Counter
from misaki import de
from kokoro_symbols import TextCleaner, dicts
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from hf_pins import rev
from common import DEV, split, load, norm
D = "data/tts"; W = f"{D}/wav"; os.makedirs(W, exist_ok=True)
GATE = os.environ["TTS_GATE_MODEL"]; MAX_WER = float(os.environ["TTS_GATE_WER"])
train, test = split()
rows = [(r, False) for r in train if r["lang"] == "de"] + [(r, True) for r in test if r["lang"] == "de"]
cache = {(c["id"], c["mtime"]): c for c in load(f"{D}/gate.jsonl") if c["model"] == GATE}
proc = model = None
g2p, clean, why, note, keep = de.DEG2P(), TextCleaner(), Counter(), Counter(), {False: [], True: []}
secs = 0.0

def to24k(r):
    wav = f"{W}/{r['id']:05d}.wav"
    if not os.path.exists(wav) or os.path.getmtime(wav) < os.path.getmtime(r["raw"]):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", r["raw"], "-ac", "1", "-ar", "24000", f"{D}/tmp.wav"], check=True)
        a, _ = sf.read(f"{D}/tmp.wav", dtype="float32")
        sf.write(wav, librosa.effects.trim(a, top_db=40)[0], 24000, subtype="PCM_16")
    return wav

def gate(r, a):
    global proc, model
    key = (r["id"], os.path.getmtime(r["raw"]))
    if key not in cache:
        if model is None:
            proc = WhisperProcessor.from_pretrained(GATE, revision=rev(GATE)); model = WhisperForConditionalGeneration.from_pretrained(GATE, revision=rev(GATE)).to(DEV).eval()
        f = proc(librosa.resample(a, orig_sr=24000, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
        with torch.no_grad(): hyp = proc.batch_decode(model.generate(f, language="de", task="transcribe"), skip_special_tokens=True)[0]
        cache[key] = {"id": r["id"], "mtime": key[1], "model": GATE, "hyp": hyp, "wer": jiwer.wer(" ".join(norm(r["text"])), " ".join(norm(hyp)))}
        with open(f"{D}/gate.jsonl", "a", encoding="utf8") as fo: fo.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    return cache[key]["wer"]

for r, held in rows:
    if not r.get("raw") or not os.path.exists(r["raw"]): why["no full-rate original (recorded before this version)"] += 1; continue
    wav = to24k(r); a, _ = sf.read(wav, dtype="float32"); dur = len(a) / 24000
    if not 2 <= dur <= 30: why["duration outside 2-30 s"] += 1; continue
    if gate(r, a) > MAX_WER: why[f"Whisper WER above {MAX_WER}"] += 1; continue
    ipa = g2p(r["text"])[0].replace("\u028f", "y")
    if len(clean(ipa)) > 510 or len(ipa) < 5: why["phoneme length"] += 1; continue
    unknown = {c for c in ipa if c not in dicts}
    if unknown: note["kept, symbols dropped by the Kokoro vocab: " + "".join(sorted(unknown))] += 1
    keep[held].append(f"{os.path.basename(wav)}|{ipa}|0"); secs += dur

tr, va = keep[False], keep[True]
if not va: n = max(1, len(tr) // 20); tr, va = tr[:-n], tr[-n:]
open(f"{D}/train_list.txt", "w", encoding="utf8").write("\n".join(tr) + "\n")
open(f"{D}/val_list.txt", "w", encoding="utf8").write("\n".join(va) + "\n")
rep = {"german_rows": len(rows), "train": len(tr), "val": len(va), "minutes": round(secs / 60, 1), "rejected": dict(why), "notes": dict(note)}
json.dump(rep, open(f"{D}/report.json", "w"), indent=1, ensure_ascii=False)
print("tts data:", json.dumps(rep, ensure_ascii=False))
if not tr: raise SystemExit("no usable German clips for TTS. see data/tts/report.json and data/tts/gate.jsonl")
if len(tr) < 300: print(f"warning: {len(tr)} training clips; the published kikiri voices used 455-627")
