import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = "ENGINE=whisper\nPARAKEET_MODEL=\nWHISPER_MODEL=base\nTORCH_BACKEND=cpu\nLANGS={langs}\nFLEURS_CFGS=en_us\nREPLAY_CFGS=\nREPLAY_N=0\nN_PROMPTS=10\nMIX_PUBLIC=anchor\nACFT_PUBLIC=y\nDOWNLOAD_DIR=./downloads\nOUT_DIR=./out\n"


def run(tmp_path, *args, langs="en", extra="TTS_LANG=de\n"):
    shutil.copy(os.path.join(ROOT, "run.sh"), tmp_path)
    (tmp_path / "config.env").write_text(CONFIG.format(langs=langs) + extra)
    return subprocess.run(["bash", str(tmp_path / "run.sh"), *args], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def test_unknown_workflow(tmp_path):
    r = run(tmp_path, "bogus")
    assert r.returncode == 1 and "unknown workflow 'bogus'" in r.stdout


def test_tts_needs_german(tmp_path):
    r = run(tmp_path, "tts")
    assert r.returncode == 1 and "TTS_LANG=de: add de to LANGS" in r.stdout
    assert "RUN_LAST=tts" in (tmp_path / "config.env").read_text()


def test_tts_language_defaults_to_english_without_german(tmp_path):
    r = run(tmp_path, "tts", langs="fr", extra="")
    assert r.returncode == 1 and "TTS_LANG=en: add en to LANGS" in r.stdout
    assert "TTS_LANG=en" in (tmp_path / "config.env").read_text().splitlines()


def test_tts_language_unsupported(tmp_path):
    r = run(tmp_path, "tts", extra="TTS_LANG=fr\n")
    assert r.returncode == 1 and "TTS_LANG=fr is not supported" in r.stdout


def test_run_last_is_replaced(tmp_path):
    run(tmp_path, "tts")
    (tmp_path / "config.env").write_text((tmp_path / "config.env").read_text().replace("RUN_LAST=tts", "RUN_LAST=stt"))
    subprocess.run(["bash", str(tmp_path / "run.sh"), "tts"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    lines = (tmp_path / "config.env").read_text().splitlines()
    assert [line for line in lines if line.startswith("RUN_LAST=")] == ["RUN_LAST=tts"] and "TORCH_BACKEND=cpu" in lines
