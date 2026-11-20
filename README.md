# voicetune

voicetune fine-tunes speech models on your own voice, on your own machine. It covers both directions:

- **Speech-to-text (STT):** fine-tunes NVIDIA Parakeet or OpenAI Whisper on your recordings and exports a quantized model for offline voice input apps on Android.
- **Text-to-speech (TTS):** trains a Kokoro-82M voice in German or English from the same recordings, using the kikiri-tts recipe.

It is a set of shell and Python scripts, not a compiled application. You start it from a terminal with `./run.sh`. A local web page shows sentences to read aloud and records them. The recordings are used for either workflow, or for both.

Training uses an AMD GPU (ROCm) or an NVIDIA GPU (CUDA) when one is available, and falls back to the CPU.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Recording](#recording)
- [Speech-to-text](#speech-to-text)
- [Text-to-speech](#text-to-speech)
- [Hardware](#hardware)
- [Configuration](#configuration)
- [Output files](#output-files)
- [Resuming and redoing steps](#resuming-and-redoing-steps)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Versioning](#versioning)
- [License](#license)

## Requirements

- Linux
- `python3`
- A browser with microphone access, on the same machine or on a phone in the same network
- Optional: an AMD GPU with ROCm, or an NVIDIA GPU with CUDA
- Optional for TTS: `espeak-ng` from your package manager (a bundled copy is used otherwise, see [Text-to-speech](#text-to-speech))

Everything else is installed inside the project folder or the download folder. Tools already on the system are used when present:

| Tool | Used when present | Fallback |
|---|---|---|
| uv | system `uv` | uv 0.13.9, pinned (all platforms) |
| Python packages | | one virtual environment per engine (`.venv-parakeet`, `.venv-whisper`, `.venv-tts`) |
| Hugging Face CLI | system `hf` | the `hf` in the virtual environment |
| ffmpeg | system `ffmpeg` | `static-ffmpeg` uv tool |
| openssl | system `openssl` | certificate made with Python `cryptography` |
| git | system `git` | GitHub tarball |
| cmake, make | system tools | `cmake` and `ninja` as uv tools |
| C++ compiler | system `c++`, `g++` or `clang++` | `zig` via the `ziglang` uv tool |

The default Hugging Face models and the FLEURS dataset are downloaded at the commits listed in `lib/hf_revisions.txt`. A model id that is not listed there loads its latest revision.

The export step builds whisper.cpp or transcribe.cpp. For a GPU build it needs the Vulkan SDK (`glslc` and headers) or ROCm. Without them it builds for the CPU.

## Quick start

```sh
git clone https://github.com/catch-26/voicetune.git
cd voicetune
./run.sh
```

Or download `voicetune-<id>.zip` from the releases page and unpack it.

`run.sh` is the only entry point. It installs what is missing, then runs the selected workflow:

```sh
./run.sh stt
./run.sh tts
./run.sh both
```

Without an argument it asks which workflow to run. The default is the last choice.

The first run asks for every setting, each with a default, and saves the answers to `config.env`. Press Enter to accept a default. Edit or delete `config.env` to change the settings later.

## How it works

1. **Prompts.** `record/stage0_data.py` picks sentences to read in your languages. With TTS selected, `tts/tts_prompts.py` adds sentences in the TTS language.
2. **Recording.** `record/stage1_record.py` serves the recorder page over HTTPS. You read the sentences aloud. It stops when the target count is reached.
3. **Training.** The STT engine or the TTS model is fine-tuned on your recordings.
4. **Evaluation.** The result is checked against held-out recordings and public speech data.
5. **Export.** The model is converted and written to the output folder.

Both workflows share `config.env`, the recorder, `data/`, the download folder and the output folder. Recordings made for one workflow are reused by the other.

## Recording

When recording starts, the terminal prints a URL such as `https://<ip>:8443/?k=<code>`. The code is new on each start; the recorder refuses requests without it.

1. Open the URL in a browser, on the same machine or on a phone in the same network.
2. Accept the warning about the self-signed certificate.
3. Do the microphone check and listen to the playback.
4. Read each sentence aloud.

With several languages, the prompts alternate between them and show the language tag. You can stop at any time. Run `./run.sh` again to continue where you stopped.

At about 5.5 seconds per sentence, 400 sentences take about 35 minutes and 650 sentences take about 1 hour.

For TTS, the recorder keeps each original upload in `data/raw/<id>.webm` and asks the browser for unprocessed audio (no echo cancellation, noise suppression or automatic gain) at 128 kbps.

### Your own sentences

Optional input files per language `<xx>`:

- `data/vocab.<xx>.txt`: your own sentences, for example with names or technical terms.
- `data/sentences_raw.<xx>.txt` or a Common Voice `data/sentences_raw.<xx>.tsv`: replaces FLEURS as the source of prompts.

## Speech-to-text

Two engines are available.

| Engine | Training | Export | Target app |
|---|---|---|---|
| `parakeet` (default) | NeMo fine-tune | transcribe.cpp GGUF, Q4_K_M (Q8_0 for the 110m model) | Offline Voice Input: Manage speech models, then import |
| `whisper` | transformers fine-tune, then ACFT | whisper.cpp GGML q8_0 | FUTO Keyboard: voice input, then import model |

### Parakeet models

| Model | Languages |
|---|---|
| `nvidia/parakeet-tdt-0.6b-v3` (default) | 25 European languages |
| `nvidia/parakeet-tdt-0.6b-v2` | English |
| `nvidia/parakeet-tdt_ctc-110m` | English, 135 MB, suited to phones |
| `nvidia/parakeet-ctc-0.6b` | English |

A local `.nemo` file also works, for example an earlier fine-tune. Plain RNN-T models are not offered because their NeMo loss runs only on NVIDIA GPUs.

In Offline Voice Input, set the language hint (for example `en` or `de`). The v3 model has no automatic language detection in transcribe.cpp.

NeMo's TDT loss needs CUDA. On ROCm and on the CPU, `stt/tdt_loss.py` replaces it with a vectorized PyTorch version that matches NeMo's reference loss. It does not apply the FastEmit and omega regularizers.

### Whisper models

`tiny.en` (English only), `tiny`, `base` (default) and `small`.

### Languages

The default is `en,de`. One language, for example `LANGS=de`, also works. English-only models set `LANGS=en`.

### Public data in the fine-tune

`MIX_PUBLIC` controls how much public FLEURS audio is mixed into the fine-tune:

- `none`: your voice only.
- `anchor` (default): your voice plus about 200 FLEURS clips per language, so that other speakers stay well recognized.
- `full`: all FLEURS audio for your languages, to retune the base model to those languages. Set `N_PROMPTS=0` to skip recording.

Other languages can be protected from forgetting with `REPLAY_CFGS` (default `es_419,it_it,ru_ru`).

### Quality gates

After the fine-tune, three checks are written to `models/eval.json`:

1. The word error rate (WER) on your own held-out recordings must improve.
2. The WER for other speakers of your languages must not get worse by more than `FORGET_TOL`.
3. The WER for the protected languages must not get worse by more than `FORGET_TOL`.

If a check fails, the run stops. `FORCE=1 ./run.sh` continues anyway. The WER treats `ß` and `ss` as equal, because the Parakeet v3 tokenizer writes `ss`.

## Text-to-speech

The TTS workflow fine-tunes Kokoro-82M on your voice with the kikiri-tts recipe. `TTS_LANG` sets the language of the voice: `de` (German) or `en` (American English). It must be one of the languages in `LANGS`. The first TTS run asks for it; the default is `de` when `LANGS` contains `de`, else `en`.

| `TTS_LANG` | Prompts | Phonemizer | Default base |
|---|---|---|---|
| `de` | Common Voice German sentences | misaki German G2P (espeak-ng), `ʏ` mapped to `y` | kikiri German base |
| `en` | Common Voice English sentences | misaki English G2P (lexicon and spaCy, espeak-ng for unknown words) | Kokoro-82M v1.0 |

The clip check and the final evaluation run Whisper in the same language.

Pinned upstream sources:

| Repository | Commit |
|---|---|
| semidark/kikiri-tts | `a12d041` |
| semidark/StyleTTS2 | `b1956da` |
| semidark/kokoro | `b96fef9` |

Steps:

1. **Prompts.** `tts/tts_prompts.py` adds sentences in `TTS_LANG` from the Common Voice sentence collector (CC0) until `TTS_CLIPS` is reached. It picks sentences that cover common phoneme pairs of that language. The recorder asks for them after the STT prompts.
2. **Preparation.** `tts/tts_prep.py` converts each original recording to 24 kHz mono, trims silence and keeps clips of 2 to 30 seconds. A stock Whisper model checks each clip and drops clips above `TTS_GATE_WER`. Phonemes come from the misaki G2P for `TTS_LANG`. For German, `ʏ` is mapped to `y` because Kokoro has no `ʏ`. For English, it also writes unrecorded Common Voice sentences as the out-of-domain texts for stage 2; German uses the kikiri-tts list. The report is written to `data/tts/report.json`.
3. **Training.** `tts/tts_train.py` trains stage 1 on your clips (this trains the style encoder that the base model lacks), then stage 2. It then extracts the voicepack, synthesizes test sentences and measures their WER with Whisper (`models/tts/eval.json`).

Base weights (`TTS_BASE`, default by `TTS_LANG`):

- `kikiri-tts/kikiri-german-base-51speakers-synthetic` (default for `de`): German stage-1 base, Apache-2.0, trained on synthetic audio of undisclosed origin.
- `hexgrad/Kokoro-82M/kokoro-v1_0.pth` (default for `en`): the original Kokoro-82M weights, Apache-2.0.
- A local `.pth` file.

Clips recorded before the recorder kept original uploads have no original file. The TTS run offers to record them again.

To switch `TTS_LANG` later, also set `TTS_BASE` to the matching base and delete `data/tts` and `models/tts`.

### espeak-ng

Kokoro reads phonemes, not text. The misaki G2P turns text into phonemes: German goes through espeak-ng (plus a small override list), English uses espeak-ng for words missing from its lexicon. misaki loads espeak-ng through the `espeakng-loader` package.

- **System copy:** on Linux, when `espeak-ng` is installed, `run.sh` links the system library and its data into `espeakng-loader`.
- **Bundled copy:** otherwise the library and data inside the `espeakng-loader` wheel from PyPI are used (`espeakng-loader==0.2.4`, espeak-ng 1.52.0), pinned in `tts/requirements.txt` like the other packages.

The espeak-ng version is written to `<out>/kokoro-<lang>-<voice>.g2p.txt`. A G2P that uses the voice elsewhere must use the same version and, for German, map `ʏ` to `y`. The English spaCy model `en_core_web_sm` 3.8.0 is pinned by URL and sha256 in `tts/requirements.txt`.

## Hardware

Training runs on PyTorch. The order is:

1. ROCm GPU (AMD) or CUDA GPU (NVIDIA), depending on `TORCH_BACKEND`.
2. CPU. This is slow but works.

`TORCH_BACKEND` is asked on the first run: `rocm7.2` (AMD, default), `cu128` (NVIDIA), `cpu` or `auto`. `DEVICE=cpu ./run.sh` forces the CPU.

With several GPUs, the first run lists them and asks which one to use. The answer is saved as `GPU=` in `config.env` and exported as `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`. Delete that line to be asked again.

The whisper.cpp and transcribe.cpp tools (quantization and one test transcription) are built for Vulkan, else HIP (ROCm, architecture from `rocminfo`), else the CPU. A failed build falls through to the next option.

TTS limits on ROCm, from kikiri-tts: fp32, batch size 4 (`TTS_BATCH`), one GPU. The first ROCm run compiles kernels for 10 to 30 minutes before the first training step.

To change the PyTorch build, edit `TORCH_BACKEND` in `config.env`, delete `.venv-<engine>` and run again.

## Configuration

All settings are in `config.env`. The main ones:

| Setting | Default | Meaning |
|---|---|---|
| `TORCH_BACKEND` | `rocm7.2` | PyTorch build: `rocm7.2`, `cu128`, `cpu`, `auto` |
| `ENGINE` | `parakeet` | STT engine: `parakeet` or `whisper` |
| `PARAKEET_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Parakeet model id or local `.nemo` path |
| `WHISPER_MODEL` | `base` | Whisper size |
| `LANGS` | `en,de` | Languages you record, ISO codes |
| `FLEURS_CFGS` | `en_us,de_de` | Matching FLEURS configs |
| `REPLAY_CFGS` | `es_419,it_it,ru_ru` | Other languages to protect from forgetting |
| `REPLAY_N` | `150` | Clips per protected language |
| `N_PROMPTS` | `400` | Sentences to record for STT |
| `MIX_PUBLIC` | `anchor` | `none`, `anchor` or `full` |
| `ACFT_PUBLIC` | `y` | Use FLEURS for the Whisper ACFT step |
| `DOWNLOAD_DIR` | `./downloads` | Models, datasets and third-party sources |
| `OUT_DIR` | `./out` | Finished models |
| `HOLDOUT` | `0.15` | Share of recordings held out for evaluation |
| `PORT` | `8443` | Recorder port |
| `FT_EPOCHS` | `3` | STT fine-tune epochs |
| `ACFT_EPOCHS` | `8` | Whisper ACFT epochs |
| `FORGET_TOL` | `0.10` | Allowed WER increase in the quality gates |
| `TTS_LANG` | `de` if in `LANGS`, else `en` | TTS voice language: `de` or `en` |
| `TTS_CLIPS` | `600` | Sentences to record for TTS in total, in `TTS_LANG` |
| `TTS_VOICE` | `myvoice` | Voice name for the output files |
| `TTS_BASE` | by `TTS_LANG` | TTS base weights |
| `TTS_TRUST_BASE` | unset | `1` loads a base checkpoint that is not plain weights (full pickle). Only for files you trust |
| `TTS_GATE_MODEL` | `openai/whisper-large-v3-turbo` | Whisper model for the clip check |
| `TTS_GATE_WER` | `0.15` | Maximum WER for a clip to be kept |
| `TTS_EPOCHS_1`, `TTS_EPOCHS_2` | `10`, `10` | TTS stage 1 and stage 2 epochs |
| `TTS_BATCH` | `4` | TTS batch size |

To switch the STT model or engine, edit `config.env`, then delete `models/` and the output file.

## Output files

| Workflow | File |
|---|---|
| STT, Parakeet | `<out>/<model>-<langs>-ft-Q4_K_M.gguf` |
| STT, Whisper | `<out>/whisper-<size>-<langs>-acft-q8_0.bin` |
| TTS | `<out>/kokoro-<lang>-<voice>.pth`, `<out>/<voice>.pt`, `<out>/kokoro-<lang>-<voice>.config.json`, `<out>/kokoro-<lang>-<voice>.g2p.txt` |

## Resuming and redoing steps

`run.sh` skips every step whose result already exists. Recording progress is kept in `data/manifest.jsonl`. Delete a result to redo that step.

| To redo | Delete |
|---|---|
| All settings | `config.env` |
| Parakeet fine-tune | `models/parakeet` |
| Whisper fine-tune | `models/ft` |
| Whisper ACFT | `models/acft` |
| TTS preparation | `data/tts` |
| TTS training | `models/tts` (`stage1.done` and `stage2.done` mark finished stages) |
| Export | the output file |
| Python packages | `.venv-<engine>` |
| Tool fallbacks | `.tools` |
| whisper.cpp or transcribe.cpp build | `<download folder>/third_party/<repo>-<commit>/build-*` |

## Troubleshooting

**GPU not visible to PyTorch.** `run.sh` prints the PyTorch version and the GPU it found. If no GPU is found:

- The PyTorch version must end in `+<TORCH_BACKEND>`. If not, delete the virtual environment and run again.
- Check `ls -l /dev/kfd /dev/dri`. You need read and write access: add your user to the `video` and `render` groups and log in again.
- For RX 7600, 7700 and 7800 cards, run `export HSA_OVERRIDE_GFX_VERSION=11.0.0` and run again.

**German or English phonemizer failed.** Install `espeak-ng` with your package manager and run again.

**TTS_LANG not in LANGS.** Add the `TTS_LANG` code to `LANGS` and its FLEURS config (`de_de` or `en_us`) to `FLEURS_CFGS` in `config.env`.

## Development

Repository layout:

| Folder | Contents |
|---|---|
| `record/` | prompt selection, the recorder server and its page |
| `stt/` | Parakeet and Whisper training, evaluation and export, with their requirements |
| `tts/` | Kokoro prompt selection, preparation and training, with its requirements |
| `lib/` | helpers shared by both workflows (`common.py`, `fetch.py`) |
| `tools/` | release tooling |
| `tests/` | tests |

`run.sh` runs every script from the repository root, with `lib/` on `PYTHONPATH`.

- Tests: `tests/`, pytest, CPU only. Install with `pip install -r tests/requirements.txt`.
- Lint: `shellcheck` for the shell scripts and `ruff` for Python.
- CI runs lint and tests on every pull request.

The recorder page is `record/recorder.html`. `record/stage1_record.py` serves it and fills in `__MIC__` and `__REC__` for the selected workflow.

## Versioning

Each release has a 5-character id (`[a-z0-9]`) derived from a hash of the shipped files. It is stored in `VERSION` and used in the archive name `voicetune-<id>.zip`.

- `python3 tools/version.py` prints the id.
- `python3 tools/version.py --write` stores it in `VERSION`.

The hash covers every tracked file except `VERSION` and the paths marked `export-ignore` in `.gitattributes` (CI, tests, tools, CHANGELOG). The same files give the same id; any change gives a new one. Changes are listed in `CHANGELOG.md`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
