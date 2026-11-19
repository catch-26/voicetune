# Changelog

Each release has a section headed `# <id> (<YYYY-MM-DD>)`, newest at the top, where `<id>` is the value in VERSION.

# Unreleased

- Without a system ffmpeg, `run.sh` now installs the `static-ffmpeg` uv tool instead of the `imageio-ffmpeg` package.
- The release archive is now named `voicetune-<id>.zip`.
- The recorder page is now a separate file, `record/recorder.html`.
- The recorder rejects uploads with an unknown sentence id or a size above 64 MiB, and `key.pem` is now readable only by its owner.
- The recorder URL now ends in `?k=<code>`, a random access code printed in the terminal; other devices on the network get no access without it.
- `tts/tts_train.py` loads the TTS base checkpoint in PyTorch's weights-only mode. A file that needs full pickle loading stops with a message; set `TTS_TRUST_BASE=1` in `config.env` only for a file you trust.
- Downloaded archives are unpacked with Python's safe tar filter and must come over HTTPS.
- `run.sh` now quotes values with spaces or shell characters when it writes `config.env`, so a download or output folder with a space works.
- Scripts and requirements files moved into the folders `record/`, `stt/`, `tts/` and `lib/`. `./run.sh` is still the only entry point.
