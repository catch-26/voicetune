"""Release id of the shipped files.

The id is 5 characters [a-z0-9] derived from a hash of every tracked file that
ships in the release archive: files marked export-ignore in .gitattributes and
VERSION itself are left out. Same files give the same id; any change gives a new one.

  python3 tools/version.py            print the id
  python3 tools/version.py --write    write it to VERSION
  python3 tools/version.py --check    fail when VERSION differs from the id
  python3 tools/version.py --archive out.zip   zip the shipped files under voicetune-<id>/
"""
import argparse
import hashlib
import os
import subprocess
import sys
import zipfile

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def git(root, *args, stdin=None):
    return subprocess.run(["git", "-C", root, *args], input=stdin, capture_output=True, check=True).stdout


def shipped(root):
    files = [f for f in git(root, "ls-files", "-z").decode().split("\0") if f]
    files = [f for f in files if os.path.isfile(os.path.join(root, f))]
    attrs = git(root, "check-attr", "-z", "--stdin", "export-ignore", stdin="\0".join(files).encode()).decode().split("\0")
    ignored = {attrs[i] for i in range(0, len(attrs) - 2, 3) if attrs[i + 2] == "set"}
    return sorted(f for f in files if f not in ignored)


def release_id(root):
    lines = "".join(
        f"{hashlib.sha256(open(os.path.join(root, f), 'rb').read()).hexdigest()}  ./{f}\n"
        for f in shipped(root) if f != "VERSION"
    )
    n = int(hashlib.sha256(lines.encode()).hexdigest(), 16)
    return "".join(ALPHABET[n // 36 ** i % 36] for i in range(5))


def archive(root, out, rid):
    prefix = f"voicetune-{rid}/"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in shipped(root):
            path = os.path.join(root, f)
            info = zipfile.ZipInfo(prefix + f, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o100755 if os.access(path, os.X_OK) else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            data = (rid + "\n").encode() if f == "VERSION" else open(path, "rb").read()
            z.writestr(info, data)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--archive")
    a = p.parse_args(argv)
    rid = release_id(a.root)
    vfile = os.path.join(a.root, "VERSION")
    if a.write:
        open(vfile, "w").write(rid + "\n")
    elif a.check:
        have = open(vfile).read().strip() if os.path.exists(vfile) else ""
        if have != rid:
            sys.exit(f"VERSION is {have!r}, the shipped files give {rid!r}: run python3 tools/version.py --write")
    elif a.archive:
        archive(a.root, a.archive, rid)
    print(rid)


if __name__ == "__main__":
    main()
