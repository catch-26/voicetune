# futo-ft: personal whisper finetune -> FUTO Keyboard, and/or an OmniVoice clone of your voice

run: `./run.sh` (the only entry point; it installs, then runs). Linux. AMD (ROCm) first, CPU fallback; NVIDIA also works.

modes (MODE in config.env, asked first):
- stt: record -> whisper finetune -> ACFT -> FUTO model
- voice: record (~20 sentences) -> OmniVoice voice clone
- both: one recording session feeds both; the voice clone runs first, so a failed finetune gate does not block it
config.env files from before modes existed have no MODE and run as stt; add MODE=both to get the clone too.

bootstrap (all inside this folder or the download folder; system tools are used when present):
- uv: system uv, else the official release binary into .tools/bin (fetch.py, needs only python3)
- python 3.13: uv provides it if the system has none
- one venv .venv for everything: torch/torchaudio from the TORCH_BACKEND index (rocm7.2 for AMD), transformers>=5.3
  (OmniVoice needs it; stt runs on it too), omnivoice only when MODE is voice or both
- hf: system hf, else the one in .venv; models are prefetched into <download folder>/hf
- ffmpeg: system, else the static imageio-ffmpeg build linked to .tools/bin/ffmpeg
- openssl: system, else the cert is made with python cryptography
- stt export only: git (else GitHub tarball), cmake and make (else cmake + ninja as uv tools),
  C++ compiler (else zig via the ziglang uv tool, wrappers in .tools/bin)
changing the torch build: edit TORCH_BACKEND, rm -rf .venv, rerun.

GPU: finetune, ACFT, reference selection and OmniVoice run on the GPU torch sees (ROCm shows as cuda);
run.sh prints the torch/hip version and asks before falling back to CPU. DEVICE=cpu ./run.sh forces CPU.
whisper.cpp (quantize + one sanity transcription) is built for CPU; it does not need the GPU.

first run asks every setting with a default and writes config.env; delete it to be asked again.
- whisper size (WHISPER_MODEL): tiny.en (39M, English only, language fixed to en, no forgetting check),
  tiny (39M), base (74M, default), small (244M). switching: edit config.env, delete models/ft models/acft and the out file
- recording: open the https URL on the phone, accept the cert warning, mic check with playback, then read; prompts alternate
  between your languages and show the language tag; recorder exits at N_PROMPTS; rerun ./run.sh anytime, it resumes
- finetune modes (MIX_PUBLIC): none = your voice only; anchor = your voice + ~200 FLEURS clips per language so other
  speakers stay good (default); full = all FLEURS audio, for retuning the base model to languages (set N_PROMPTS=0 to skip recording)
- three gates after the finetune (models/eval.json): your own held-out WER must improve, other speakers of your languages
  and the protected languages must not get worse by more than FORGET_TOL; `FORCE=1 ./run.sh` continues anyway
- stt output: <out folder>/whisper-<size>-<langs>-acft-q8_0.bin -> copy to phone -> FUTO Keyboard voice input -> import model

voice clone:
- per language the recording closest to 8 s whose whisper-base transcript matches the prompt best becomes the reference;
  it is cut from the original phone audio in data/raw (full band, 24 kHz); recordings made before data/raw existed fall
  back to the 16 kHz data/wav copy
- output: <out folder>/voice.<xx>.pt (reusable OmniVoice prompt) and voice.<xx>.sample.wav to listen to;
  data/voice/<xx>.wav + <xx>.txt are the reference pair (also usable as reference audio/text in other zero-shot TTS)
- redo: delete <out folder>/voice.*.pt (keeps the chosen reference) or also data/voice/ (picks again)

optional inputs per language <xx>: data/vocab.<xx>.txt (your own sentences with names/jargon),
data/sentences_raw.<xx>.txt or a Common Voice data/sentences_raw.<xx>.tsv to replace FLEURS as prompt source.

state: data/manifest.jsonl. delete models/ft to retrain, models/acft to redo ACFT, the out folder to re-export,
.venv to reinstall python deps, .tools to redo the tool fallbacks.

versioning: every release gets a 5-character id [a-z0-9] derived from a hash of the release files. it is stored in
VERSION and appended to the archive name: futo-ft-<id>.zip. compute it from the folder root after all changes, with
VERSION itself and local state (.venv, .tools, data, models, downloads, out, config.env, *.pem) excluded:
`find . -type f ! -name VERSION ! -path './.venv/*' ! -path './.tools/*' ! -path './data/*' ! -path './models/*' ! -path './downloads/*' ! -path './out/*' ! -name config.env ! -name '*.pem' | sort | xargs sha256sum | sha256sum | python3 -c "import sys; n=int(sys.stdin.read()[:64],16); a='0123456789abcdefghijklmnopqrstuvwxyz'; print(''.join(a[n//36**i%36] for i in range(5)))" > VERSION`
same files give the same id; any change gives a new one.
