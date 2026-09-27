import json, os, subprocess, torch, jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import DEV, load, audio, norm
BASE = "openai/whisper-base"
proc = WhisperProcessor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(BASE).to(DEV).eval()
os.makedirs("data/voice", exist_ok=True)

def dur(r):
    a, sr = audio(r); return len(a) / sr

def wer(r):
    a, sr = audio(r)
    f = proc(a, sampling_rate=sr, return_tensors="pt").input_features.to(DEV)
    with torch.no_grad(): ids = model.generate(f, language=r["lang"], task="transcribe")
    return jiwer.wer(" ".join(norm(r["text"])), " ".join(norm(proc.batch_decode(ids, skip_special_tokens=True)[0])))

refs = {}
rows = load("data/manifest.jsonl")
for lang in sorted({r["lang"] for r in rows}):
    rs = [r for r in rows if r["lang"] == lang]
    cand = sorted(((dur(r), r) for r in rs), key=lambda x: abs(x[0] - 8))[:30]
    w, d, best = min(((wer(r), d, r) for d, r in cand), key=lambda x: (x[0], abs(x[1] - 8)))
    raw = f"data/raw/{best['id']:05d}.webm"
    src = raw if os.path.exists(raw) else best["wav"]
    wav = f"data/voice/{lang}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "24000", wav], check=True)
    open(f"data/voice/{lang}.txt", "w", encoding="utf8").write(best["text"])
    sample = max((r["text"] for r in rs if r["id"] != best["id"]), key=len, default=best["text"])
    refs[lang] = {"id": best["id"], "wav": wav, "text": best["text"], "wer": w, "seconds": round(d, 1), "source": src, "sample": sample}
    print(f"{lang}: reference clip {best['id']} ({refs[lang]['seconds']} s, whisper WER {refs[lang]['wer']:.2f}, from {src})")
json.dump(refs, open("data/voice/refs.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
