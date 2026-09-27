import json, os, soundfile as sf, torch
from omnivoice import OmniVoice
dev = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dev, dtype=torch.float32 if dev == "cpu" else torch.float16)
out = os.environ["OUT_DIR"]
for lang, r in json.load(open("data/voice/refs.json", encoding="utf8")).items():
    prompt = model.create_voice_clone_prompt(ref_audio=r["wav"], ref_text=r["text"])
    sf.write(f"{out}/voice.{lang}.sample.wav", model.generate(text=r["sample"], language=lang, voice_clone_prompt=prompt)[0], model.sampling_rate)
    prompt.save(f"{out}/voice.{lang}.pt")
    print(f"{lang}: {out}/voice.{lang}.pt, listen: {out}/voice.{lang}.sample.wav")
