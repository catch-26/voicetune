import importlib
import json

import pytest

ENV = {"LANGS": "en,de", "FLEURS_CFGS": "en_us,de_de", "REPLAY_CFGS": "es_419,", "HOLDOUT": "0.15", "ANCHOR_EVAL": "2",
       "ENGINE": "whisper", "WHISPER_MODEL": "base", "PARAKEET_MODEL": "", "DEVICE": "cpu"}


@pytest.fixture
def common(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    import common
    return importlib.reload(common)


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")


def test_config(common):
    assert common.LANGS == ["en", "de"]
    assert common.REPLAY_CFGS == ["es_419"]
    assert common.BASE == "openai/whisper-base"
    assert common.DEV == "cpu"
    assert common.lang_kw("de") == {"language": "de", "task": "transcribe"}


def test_english_only_model(common, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny.en")
    c = importlib.reload(common)
    assert c.EN_ONLY and c.lang_kw("en") == {}


def test_parakeet_base(common, monkeypatch):
    monkeypatch.setenv("ENGINE", "parakeet")
    monkeypatch.setenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    assert importlib.reload(common).BASE == "nvidia/parakeet-tdt-0.6b-v3"


def test_lang_of(common):
    assert common.lang_of("de_de") == "de"
    assert common.lang_of("cmn_hans_cn") == "zh"
    assert common.lang_of("fil_ph") == "tl"
    assert common.lang_of("es_419") == "es"


def test_norm(common):
    assert common.norm("Die Straße, bitte!") == ["die", "strasse", "bitte"]
    assert common.norm("  Hello   World. ") == ["hello", "world"]


def test_split_is_stable(common, tmp_path):
    write(tmp_path / "data/manifest.jsonl", [{"id": i, "lang": "en", "text": "t", "wav": f"{i}.wav"} for i in range(200)])
    train, test = common.split()
    assert len(train) + len(test) == 200
    assert {r["id"] for r in test} == {i for i in range(200) if i % 100 < 15}


def test_split_without_manifest(common):
    assert common.split() == ([], [])


def test_fleurs_and_replay(common, tmp_path):
    for c in ("en_us", "de_de"):
        write(tmp_path / f"data/fleurs.{c}.jsonl", [{"id": i, "cfg": c} for i in range(5)])
    write(tmp_path / "data/replay.es_419.jsonl", [{"id": i} for i in range(6)])
    tr, ev = common.fleurs()
    assert len(tr) == 6 and len(ev) == 4
    assert [r["id"] for r in ev] == [3, 4, 3, 4]
    tr, ev = common.replay()
    assert [r["id"] for r in tr] == [0, 1, 2] and [r["id"] for r in ev] == [3, 4, 5]
