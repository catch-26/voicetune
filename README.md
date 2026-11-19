# stt-tts-ft: personal speech-to-text finetune -> phone, and a German Kokoro voice of the same recordings

run: `./run.sh [stt|tts|both]` (the only entry point; it installs, then runs). without an argument it asks, default = last choice
(RUN_LAST in config.env). both workflows share config.env, the recorder, data/, the download folder and the out folder;
recordings made for one are reused by the other. Linux. AMD first, CPU fallback; NVIDIA also works.
several GPUs: the first run lists them and asks once (GPU= in config.env, exported as HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES);
delete that line to be asked again.
record -> finetune -> quantize -> export. two engines:
- parakeet (default): NeMo finetune -> transcribe.cpp GGUF -> Offline Voice Input (notune/android_transcribe_app), Manage speech models -> import
- whisper: transformers finetune -> ACFT -> whisper.cpp GGML q8_0 -> FUTO Keyboard voice input -> import model

hardware order: training runs on torch, so ROCm GPU, else CPU (torch has no Vulkan training). the whisper.cpp / transcribe.cpp tools
(quantize + one sanity transcription) are built for Vulkan, else HIP (ROCm, arch from rocminfo), else CPU; a failed build falls
through to the next. DEVICE=cpu ./run.sh forces CPU for both.

bootstrap (all inside this folder or the download folder; system tools are used when present):
- uv: system uv, else the official release binary into .tools/bin (fetch.py, needs only python3)
- one venv per engine: .venv-parakeet (python 3.12, nemo_toolkit[asr]) or .venv-whisper (python 3.13, transformers>=5.3);
  torch from the TORCH_BACKEND index (rocm7.2 for AMD)
- hf: system hf, else the one in the venv; models are prefetched into <download folder>/hf
- ffmpeg: system, else the static imageio-ffmpeg build linked to .tools/bin/ffmpeg
- openssl: system, else the cert is made with python cryptography
- export step only: git (else GitHub tarball), cmake and make (else cmake + ninja as uv tools),
  C++ compiler (else zig via the ziglang uv tool, wrappers in .tools/bin); Vulkan SDK (glslc, headers) or ROCm for GPU builds
changing the torch build: edit TORCH_BACKEND, rm -rf .venv-<engine>, rerun.

first run asks every setting with a default and writes config.env; delete it to be asked again.
- engine (ENGINE): parakeet | whisper
- parakeet model (PARAKEET_MODEL): nvidia/parakeet-tdt-0.6b-v3 (25 European languages, default), English only:
  nvidia/parakeet-tdt-0.6b-v2, nvidia/parakeet-tdt_ctc-110m (135 MB, phone-sized), nvidia/parakeet-ctc-0.6b.
  a local .nemo path also works (e.g. an earlier finetune). export: Q4_K_M, Q8_0 for 110m.
  plain RNN-T models (rnnt-0.6b, unified-en) are not offered: their NeMo loss only runs on NVIDIA.
  in Offline Voice Input set the language hint (en, de): v3 has no auto-detect in transcribe.cpp.
- whisper size (WHISPER_MODEL): tiny.en, tiny, base (default), small
- languages: default en,de; one language (LANGS=de) is fine. English-only models fix LANGS=en
- switching model or engine: edit config.env, delete models/ and the out file
- recording: open the https URL on the phone, accept the cert warning, mic check with playback, then read; prompts alternate
  between your languages and show the language tag; recorder exits at N_PROMPTS; rerun ./run.sh anytime, it resumes
- finetune modes (MIX_PUBLIC): none = your voice only; anchor = your voice + ~200 FLEURS clips per language so other
  speakers stay good (default); full = all FLEURS audio, for retuning the base model to languages (set N_PROMPTS=0 to skip recording)
- three gates after the finetune (models/eval.json): your own held-out WER must improve, other speakers of your languages
  and the protected languages must not get worse by more than FORGET_TOL; `FORCE=1 ./run.sh` continues anyway.
  WER folds ß to ss (the parakeet v3 tokenizer writes ss)
- output: parakeet <out folder>/<model>-<langs>-ft-Q4_K_M.gguf; whisper <out folder>/whisper-<size>-<langs>-acft-q8_0.bin

parakeet TDT on AMD/CPU: NeMo's TDT loss needs NVIDIA CUDA (numba), so on ROCm and CPU tdt_loss.py (vectorized PyTorch,
matches NeMo's reference loss) replaces it. it skips the fastemit/omega training regularizers.

optional inputs per language <xx>: data/vocab.<xx>.txt (your own sentences with names/jargon),
data/sentences_raw.<xx>.txt or a Common Voice data/sentences_raw.<xx>.tsv to replace FLEURS as prompt source.

tts (German, Kokoro-82M via the kikiri-tts recipe, pinned: semidark/kikiri-tts a12d041, StyleTTS2 b1956da, kokoro b96fef9):
- needs de in LANGS. asks TTS_CLIPS (German sentences in total, default 600), TTS_VOICE, TTS_BASE
- prompts: stage0 as before, then tts_prompts.py appends German sentences from the Common Voice sentence-collector list (CC0)
  until TTS_CLIPS, picked to cover common German phoneme pairs; the recorder then asks for them after the STT prompts
- recorder: keeps each original upload in data/raw/<id>.webm (the 16 kHz wav alone is too narrow for TTS); with tts selected it
  also asks the browser for unprocessed audio (no echo cancel / noise suppression / auto gain) at 128 kbps. German clips recorded
  before this version have no original: the tts run offers to re-record them
- tts_prep.py: original -> 24 kHz mono, trim silence, keep 2-30 s, stock Whisper check (TTS_GATE_MODEL, drop above TTS_GATE_WER;
  cached in data/tts/gate.jsonl), phonemes via the misaki fork German G2P then ʏ->y (Kokoro has no ʏ), same held-out split as STT.
  report: data/tts/report.json
- tts_train.py: base (default kikiri-tts/kikiri-german-base-51speakers-synthetic, German stage-1 base, Apache-2.0, trained on
  synthetic audio of undisclosed origin; hexgrad/Kokoro-82M/kokoro-v1_0.pth is the English alternative) -> stage 1 on your clips
  (trains the style encoder the base lacks) -> stage 2 -> voicepack -> kikiri test sentences -> Whisper WER (models/tts/eval.json)
- ROCm limits from kikiri: fp32, batch 4 (TTS_BATCH), single GPU. the first ROCm run compiles kernels for 10-30 min before step 1
- phonemizer: the system espeak-ng is used when present (bundled loader symlinked to it), else the bundled one. the version is
  written next to the model; the browser G2P must match it and map ʏ->y (kikiri's KPipeline drops ʏ instead)
- output: <out>/kokoro-de-<voice>.pth, <voice>.pt, kokoro-de-<voice>.config.json, kokoro-de-<voice>.g2p.txt
- redo: data/tts to redo prep, models/tts to retrain (stage1.done/stage2.done mark finished stages), the out .pth to re-export

state: data/manifest.jsonl. parakeet: delete models/parakeet to retrain. whisper: models/ft to retrain, models/acft to redo ACFT.
the out file to re-export, .venv-<engine> to reinstall python deps, .tools to redo the tool fallbacks,
<download folder>/third_party/<repo>/build-* to rebuild the tools.

versioning: every release gets a 5-character id [a-z0-9] derived from a hash of the release files. it is stored in
VERSION and appended to the archive name: stt-tts-ft-<id>.zip. compute it from the folder root after all changes, with
VERSION itself and local state (.venv*, .tools, data, models, downloads, out, config.env, *.pem) excluded:
`find . -type f ! -name VERSION ! -path './.venv*' ! -path './.tools/*' ! -path './data/*' ! -path './models/*' ! -path './downloads/*' ! -path './out/*' ! -name config.env ! -name '*.pem' | sort | xargs sha256sum | sha256sum | python3 -c "import sys; n=int(sys.stdin.read()[:64],16); a='0123456789abcdefghijklmnopqrstuvwxyz'; print(''.join(a[n//36**i%36] for i in range(5)))" > VERSION`
same files give the same id; any change gives a new one.
