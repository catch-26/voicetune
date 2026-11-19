# Changelog

Each release has a section headed `# <id> (<YYYY-MM-DD>)`, newest at the top, where `<id>` is the value in VERSION.

# Unreleased

- Without a system ffmpeg, `run.sh` now installs the `static-ffmpeg` uv tool instead of the `imageio-ffmpeg` package.
- The release archive is now named `voicetune-<id>.zip`.
- The recorder page is now a separate file, `web/recorder.html`.
- The recorder rejects uploads with an unknown sentence id or a size above 64 MiB, and `key.pem` is now readable only by its owner.
- Downloaded archives are unpacked with Python's safe tar filter and must come over HTTPS.
- `run.sh` now quotes values with spaces or shell characters when it writes `config.env`, so a download or output folder with a space works.
