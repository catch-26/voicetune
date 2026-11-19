import os, re
TO_CUDA = re.compile(r"\.to\(\s*[\"']cuda[\"']\s*\)")
CUDA = re.compile(r"[\"']cuda[\"']")
FILES = {"train_first.py": ".to(device)", "train_second.py": ".to(device)", "Modules/hifigan.py": ".to(asr.device)", "Modules/istftnet.py": ".to(asr.device)"}
OLD, NEW = 'device = "cuda"\n', 'device = "cuda" if torch.cuda.is_available() else "cpu"\n'


def patch(st):
    for name, repl in FILES.items():
        p = os.path.join(st, name)
        src = open(p, encoding="utf8").read()
        new = TO_CUDA.sub(repl, src)
        if name == "train_second.py": new = new.replace("    " + OLD, "    " + NEW)
        if new != src: open(p, "w", encoding="utf8").write(new)
        left = [l.strip() for l in new.splitlines() if CUDA.search(l) and l.strip() != NEW.strip()]
        if left: raise RuntimeError(f"{p}: hard-coded cuda device left: {left}")
