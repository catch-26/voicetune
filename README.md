# futo-ft: personal whisper finetune -> FUTO Keyboard

run: `./run.sh` (the only entry point; it installs, then runs). Linux. AMD (ROCm) first, CPU fallback; NVIDIA also works.
record -> whisper finetune -> ACFT -> quantized FUTO model. the OmniVoice voice clone lives in its own runtime (omnivoice-rt).

bootstrap (all inside this folder or the download folder; system tools are used when present):
- uv: system uv, else the official release binary into .tools/bin (fetch.py, needs only python3)
- python 3.13: uv provides it if the system has none
- one venv .venv for everything: torch from the TORCH_BACKEND index (rocm7.2 for AMD), transformers>=5.3
  (the stage code targets the transformers 5 API)
- hf: system hf, else the one in .venv; models are prefetched into <download folder>/hf
- ffmpeg: system, else the static imageio-ffmpeg build linked to .tools/bin/ffmpeg
- openssl: system, else the cert is made with python cryptography
- export step only: git (else GitHub tarball), cmake and make (else cmake + ninja as uv tools),
  C++ compiler (else zig via the ziglang uv tool, wrappers in .tools/bin)
changing the torch build: edit TORCH_BACKEND, rm -rf .venv, rerun.

GPU: finetune and ACFT run on the GPU torch sees (ROCm shows as cuda);
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
- output: <out folder>/whisper-<size>-<langs>-acft-q8_0.bin -> copy to phone -> FUTO Keyboard voice input -> import model

optional inputs per language <xx>: data/vocab.<xx>.txt (your own sentences with names/jargon),
data/sentences_raw.<xx>.txt or a Common Voice data/sentences_raw.<xx>.tsv to replace FLEURS as prompt source.

state: data/manifest.jsonl. delete models/ft to retrain, models/acft to redo ACFT, the out folder to re-export,
.venv to reinstall python deps, .tools to redo the tool fallbacks.

versioning: every release gets a 5-character id [a-z0-9] derived from a hash of the release files. it is stored in
VERSION and appended to the archive name: futo-ft-<id>.zip. compute it from the folder root after all changes, with
VERSION itself and local state (.venv, .tools, data, models, downloads, out, config.env, *.pem) excluded:
`find . -type f ! -name VERSION ! -path './.venv/*' ! -path './.tools/*' ! -path './data/*' ! -path './models/*' ! -path './downloads/*' ! -path './out/*' ! -name config.env ! -name '*.pem' | sort | xargs sha256sum | sha256sum | python3 -c "import sys; n=int(sys.stdin.read()[:64],16); a='0123456789abcdefghijklmnopqrstuvwxyz'; print(''.join(a[n//36**i%36] for i in range(5)))" > VERSION`
same files give the same id; any change gives a new one.
