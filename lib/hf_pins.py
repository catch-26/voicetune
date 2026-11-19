import os

PINS = {}
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_revisions.txt"), encoding="utf8"):
    parts = line.split("#")[0].split()
    if len(parts) == 2:
        PINS[parts[0]] = parts[1]


def rev(repo):
    return PINS.get(repo)
