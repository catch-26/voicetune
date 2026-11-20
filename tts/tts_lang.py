import hashlib, os, urllib.request
LANG = os.environ.get("TTS_LANG", "de")
NAME = {"de": "German", "en": "English"}[LANG]
CV = f"https://raw.githubusercontent.com/common-voice/common-voice/d910b8578b57d74597c5972f217d72abb696c55a/server/data/{LANG}/sentence-collector.txt"
CV_SHA256 = {"de": "de6bd33fc5742350690fa4be6197aafd9be2c6c9fe4b70b8c5c8670971acc117", "en": "31ac8e200449ebd7aee4a8c6bd16d4dafd076cf11f7c3efe1035f0283712d136"}[LANG]
SAMPLE = {"de": "Guten Tag", "en": "Good morning"}[LANG]
TEST_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "I thought I saw three thin thieves on Thursday.",
    "Would you like to measure the treasure in the garage?",
    "Why did you do that? That is unbelievable!",
    "The weather up north is rather warm this year, isn't it?",
    "It costs exactly one hundred and twenty-three million dollars.",
]


def g2p():
    if LANG == "de":
        from misaki import de
        g = de.DEG2P()
        return lambda s: g(s)[0].replace("ʏ", "y")
    from misaki import en, espeak
    g = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
    return lambda s: g(s)[0]


def g2p_note():
    if LANG == "de": return "trained with misaki fork 9cda926 DEG2P, then ʏ -> y\n"
    return "trained with misaki fork 9cda926 en.G2P (American English, spaCy en_core_web_sm, espeak-ng fallback)\n"


def pool():
    f = os.environ["DOWNLOAD_DIR"] + f"/cv_{LANG}_sentences.txt"
    if not os.path.exists(f):
        data = urllib.request.urlopen(CV, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != CV_SHA256: raise SystemExit(f"sha256 mismatch for {CV}")
        open(f, "wb").write(data)
    lines = open(f, encoding="utf8").read().splitlines()
    own = f"data/sentences_raw.{LANG}.txt"
    if os.path.exists(own): lines += open(own, encoding="utf8").read().splitlines()
    return lines


if __name__ == "__main__":
    assert g2p()(SAMPLE)
