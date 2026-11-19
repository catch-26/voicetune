import glob
import os
import py_compile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "*", "*.py"))), ids=lambda p: os.path.relpath(p, ROOT))
def test_compiles(path, tmp_path):
    py_compile.compile(path, cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_requirements_parse():
    from packaging.requirements import Requirement
    for f in glob.glob(os.path.join(ROOT, "stt", "requirements-*.txt")) + [os.path.join(ROOT, "tts", "requirements.txt")]:
        for line in open(f, encoding="utf8"):
            if line.strip() and not line.startswith("#"):
                Requirement(line.strip())


def test_recorder_page_placeholders():
    page = open(os.path.join(ROOT, "record", "recorder.html"), encoding="utf8").read()
    assert page.count("__MIC__") == 1 and page.count("__REC__") == 1


def test_transformers_major_versions():
    from packaging.requirements import Requirement
    from packaging.version import Version

    def version(path):
        for line in open(os.path.join(ROOT, path), encoding="utf8"):
            r = Requirement(line.strip()) if line.strip() else None
            if r and r.name == "transformers":
                return Version(next(iter(r.specifier)).version)
    assert version("tts/requirements.txt").major == 4
    assert version("stt/requirements-whisper.txt") >= Version("5.3")


def test_hf_revisions_pinned():
    import re
    pins = {}
    for line in open(os.path.join(ROOT, "lib", "hf_revisions.txt"), encoding="utf8"):
        parts = line.split("#")[0].split()
        if parts:
            assert len(parts) == 2 and re.fullmatch(r"[0-9a-f]{40}", parts[1]), line
            pins[parts[0]] = parts[1]
    run = open(os.path.join(ROOT, "run.sh"), encoding="utf8").read()
    for repo in ["nvidia/parakeet-tdt-0.6b-v3", "openai/whisper-large-v3-turbo", "kikiri-tts/kikiri-german-base-51speakers-synthetic", "hexgrad/Kokoro-82M"]:
        assert repo in run and repo in pins, repo
    for size in ["tiny.en", "tiny", "base", "small"]:
        assert "openai/whisper-" + size in pins
