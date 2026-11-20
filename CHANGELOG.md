# Changelog

Each release has a section headed `# <id> (<YYYY-MM-DD>)`, newest at the top, where `<id>` is the value in VERSION.

# Unreleased

- The TTS workflow can train an English voice. The new `TTS_LANG` setting (`de` or `en`) picks the prompts, the phonemizer, the Whisper language for the clip check and evaluation, and the default base weights (Kokoro-82M v1.0 for English). German works as before; output files are now named `kokoro-<lang>-<voice>.*`.
- Pin espeakng-loader 0.2.4 (bundled espeak-ng 1.52.0), phonemizer-fork 3.3.2 and spaCy 3.8.18 for TTS, and install the spaCy model en_core_web_sm 3.8.0 from a pinned URL with sha256 instead of downloading it at run time.
- The default Hugging Face models and the FLEURS dataset are downloaded at fixed revisions, listed in `lib/hf_revisions.txt`.
- Without a system uv, `run.sh` installs uv 0.13.9 and checks its sha256 (Linux and macOS, x86_64 and arm64), instead of the latest release.
- TTS training now runs on the CPU when no GPU is available, or with `DEVICE=cpu`. `tts/tts_train.py` replaces the hard-coded `cuda` device in the pinned StyleTTS2 training scripts before training starts.
- Parakeet training uses no DataLoader worker processes on systems other than Linux, where workers are started with `spawn`.
- `run.sh` no longer uses `sed -i` or `ldconfig` outside Linux, downloads the matching uv build on macOS, and offers `cpu` as the default PyTorch build on macOS.
- The export step now downloads whisper.cpp, transcribe.cpp and openai/whisper at fixed commits (ca19af2, adfc502, 8609812) into `third_party/<repo>-<commit>/` instead of the latest commit.
- Without a system ffmpeg, `run.sh` now installs the `static-ffmpeg` uv tool instead of the `imageio-ffmpeg` package.
- The release archive is now named `voicetune-<id>.zip`.
- The recorder page is now a separate file, `record/recorder.html`.
- The recorder rejects uploads with an unknown sentence id or a size above 64 MiB, and `key.pem` is now readable only by its owner.
- The recorder URL now ends in `?k=<code>`, a random access code printed in the terminal; other devices on the network get no access without it.
- `tts/tts_train.py` loads the TTS base checkpoint in PyTorch's weights-only mode. A file that needs full pickle loading stops with a message; set `TTS_TRUST_BASE=1` in `config.env` only for a file you trust.
- Downloaded archives are unpacked with Python's safe tar filter and must come over HTTPS.
- `run.sh` now quotes values with spaces or shell characters when it writes `config.env`, so a download or output folder with a space works.
- Scripts and requirements files moved into the folders `record/`, `stt/`, `tts/` and `lib/`. `./run.sh` is still the only entry point.
