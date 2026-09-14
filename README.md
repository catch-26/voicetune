# futo-ft: personal multilingual whisper-base -> FUTO Keyboard

needs on PATH: uv, ffmpeg, openssl, git, cmake, a C++ compiler. Linux. AMD (ROCm) or CPU; NVIDIA also works.
python deps are installed by run.sh into ./.venv via uv; the torch build is chosen in the first-run questions (rocm7.2 for AMD).
changing the torch build later: edit TORCH_BACKEND in config.env, then rm -rf .venv and rerun.

run: `./run.sh`
- first run asks every setting with a default and writes config.env; delete it to be asked again
- everything downloaded (base model, FLEURS audio, whisper.cpp, openai/whisper) lands in the download folder and is reused
- recording: open the https URL on the phone, accept the cert warning, mic check with playback, then read; prompts alternate
  between your languages and show the language tag; recorder exits at N_PROMPTS; rerun ./run.sh anytime, it resumes
- finetune modes (MIX_PUBLIC): none = your voice only; anchor = your voice + ~200 FLEURS clips per language so other
  speakers stay good (default); full = all FLEURS audio, for retuning the base model to languages (set N_PROMPTS=0 to skip recording)
- three gates after the finetune (models/eval.json): your own held-out WER must improve, other speakers of your languages
  and the protected languages must not get worse by more than FORGET_TOL; `FORCE=1 ./run.sh` continues anyway
- output: <out folder>/whisper-base-en-de-acft-q8_0.bin -> copy to phone -> FUTO Keyboard voice input -> import model

optional inputs per language <xx>: data/vocab.<xx>.txt (your own sentences with names/jargon),
data/sentences_raw.<xx>.txt or a Common Voice data/sentences_raw.<xx>.tsv to replace FLEURS as prompt source.

state: data/manifest.jsonl. delete models/ft to retrain, models/acft to redo ACFT, the out folder to re-export,
.venv to reinstall python deps.
