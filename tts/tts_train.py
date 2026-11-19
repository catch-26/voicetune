import glob, importlib.util, json, os, pickle, shutil, subprocess, sys, jiwer, librosa, soundfile as sf, torch, yaml
from huggingface_hub import hf_hub_download, list_repo_files
from hf_pins import rev
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import DEV, norm
from device_patch import patch
K = os.environ["KIKIRI"]; ST = f"{K}/StyleTTS2"; M = os.path.abspath("models/tts"); LOG = f"{M}/logs"
VOICE = os.environ["TTS_VOICE"]; OUT = os.environ["OUT_DIR"]; D = os.path.abspath("data/tts")
os.makedirs(LOG, exist_ok=True)

base = f"{M}/base.pth"
if not os.path.exists(base):
    src = os.environ["TTS_BASE"]
    if not os.path.exists(src):
        repo, _, name = src.partition(".pth")[0].rpartition("/") if src.endswith(".pth") else (src, "", "")
        if not name:
            found = [f for f in list_repo_files(repo, revision=rev(repo)) if f.endswith(".pth") and "/" not in f]
            if len(found) != 1: sys.exit(f"{repo}: expected one .pth at the repo root, found {found}. set TTS_BASE=<repo>/<file>.pth in config.env")
            name = found[0].removesuffix(".pth")
        src = hf_hub_download(repo, name + ".pth", revision=rev(repo))
    try:
        state = torch.load(src, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        if os.environ.get("TTS_TRUST_BASE") != "1":
            sys.exit(f"{src} holds more than plain weights and was not loaded:\n{e}\nonly if you trust this file, add TTS_TRUST_BASE=1 to config.env and rerun")
        state = torch.load(src, map_location="cpu", weights_only=False)
    if "net" not in state: state = {"net": {k: {n.removeprefix("module."): t for n, t in v.items()} for k, v in state.items()}}
    print("base components:", list(state["net"]))
    torch.save({"net": state["net"]}, base)

cfg = yaml.safe_load(open(f"{K}/configs/config_german_ft.yml"))
E1, E2 = int(os.environ["TTS_EPOCHS_1"]), int(os.environ["TTS_EPOCHS_2"])
cfg.update(batch_size=int(os.environ["TTS_BATCH"]), epochs_1st=E1, epochs_2nd=E2, save_freq=1, pretrained_model=base, log_dir=LOG,
           first_stage_path="first_stage.pth", load_only_params=True, second_stage_load_pretrained=False)
cfg["data_params"].update(train_data=f"{D}/train_list.txt", val_data=f"{D}/val_list.txt", root_path=f"{D}/wav",
                          OOD_data=f"{K}/training/OOD_texts.txt", num_workers=min(8, os.cpu_count() or 1))
yaml.safe_dump(cfg, open(f"{M}/config.yml", "w"), allow_unicode=True, sort_keys=False)

patch(ST)
if DEV == "cpu": os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
for stage, script in [("stage1", "train_first.py"), ("stage2", "train_second.py")]:
    if os.path.exists(f"{M}/{stage}.done"): continue
    print(f"tts {stage}: {script} on {DEV} (first ROCm run compiles kernels for 10-30 min before step 1)")
    subprocess.run([sys.executable, script, "--config_path", f"{M}/config.yml"], cwd=ST, check=True)
    open(f"{M}/{stage}.done", "w").close()

last = f"{LOG}/epoch_2nd_{E2 - 1:05d}.pth"
if not os.path.exists(last): last = sorted(glob.glob(f"{LOG}/epoch_2nd_*.pth"))[-1]
vp, test = f"{M}/{VOICE}.pt", f"{M}/test"
subprocess.run([sys.executable, f"{K}/scripts/extract_voicepack.py", "--model", last, "--style-encoder-model", f"{LOG}/first_stage.pth",
                "--audio-dir", f"{D}/wav", "--output", vp], check=True)
subprocess.run([sys.executable, f"{K}/scripts/test_inference.py", "--checkpoint", last, "--voicepack", vp,
                "--config", f"{K}/training/config.json", "--output-dir", test], check=True)

spec = importlib.util.spec_from_file_location("ti", f"{K}/scripts/test_inference.py"); ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
G = os.environ["TTS_GATE_MODEL"]; proc = WhisperProcessor.from_pretrained(G, revision=rev(G)); asr = WhisperForConditionalGeneration.from_pretrained(G, revision=rev(G)).to(DEV).eval()
res = {}
for i, text in enumerate(ti.TEST_SENTENCES):
    wav = f"{test}/test_{i + 1:02d}.wav"
    if not os.path.exists(wav): res[text] = None; continue
    a, sr = sf.read(wav, dtype="float32")
    f = proc(librosa.resample(a, orig_sr=sr, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
    with torch.no_grad(): hyp = proc.batch_decode(asr.generate(f, language="de", task="transcribe"), skip_special_tokens=True)[0]
    res[text] = {"hyp": hyp, "wer": jiwer.wer(" ".join(norm(text)), " ".join(norm(hyp)))}
ok = [v["wer"] for v in res.values() if v]
json.dump({"checkpoint": last, "mean_wer": sum(ok) / max(1, len(ok)), "missing": sum(v is None for v in res.values()), "sentences": res},
          open(f"{M}/eval.json", "w"), indent=1, ensure_ascii=False)
print(f"tts intelligibility (stock Whisper on {len(ok)} test sentences): mean WER {sum(ok) / max(1, len(ok)):.3f}. listen: {test}/")

os.makedirs(OUT, exist_ok=True)
shutil.copy(f"{test}/kokoro_german_converted.pth", f"{OUT}/kokoro-de-{VOICE}.pth")
shutil.copy(vp, f"{OUT}/{VOICE}.pt")
shutil.copy(f"{K}/training/config.json", f"{OUT}/kokoro-de-{VOICE}.config.json")
open(f"{OUT}/kokoro-de-{VOICE}.g2p.txt", "w").write("trained with misaki fork 9cda926 DEG2P, then \u028f -> y\n" + (subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).stdout if shutil.which("espeak-ng") else "bundled espeakng-loader\n"))
