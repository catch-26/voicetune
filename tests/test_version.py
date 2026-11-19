import os
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import version  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    def sh(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    sh("init", "-q")
    (tmp_path / "run.sh").write_text("echo hi\n")
    (tmp_path / "run.sh").chmod(0o755)
    (tmp_path / "VERSION").write_text("old\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/t.py").write_text("x = 1\n")
    (tmp_path / ".gitattributes").write_text("/tests/** export-ignore\n/.gitattributes export-ignore\n")
    sh("add", "-A")
    return tmp_path


def test_id_ignores_version_and_export_ignore(repo):
    rid = version.release_id(str(repo))
    assert len(rid) == 5 and rid.isalnum() and rid.islower() or rid.isdigit()
    (repo / "VERSION").write_text("other\n")
    (repo / "tests/t.py").write_text("x = 2\n")
    assert version.release_id(str(repo)) == rid
    (repo / "run.sh").write_text("echo bye\n")
    assert version.release_id(str(repo)) != rid


def test_write_check_archive(repo, tmp_path_factory):
    with pytest.raises(SystemExit):
        version.main(["--root", str(repo), "--check"])
    version.main(["--root", str(repo), "--write"])
    rid = (repo / "VERSION").read_text().strip()
    version.main(["--root", str(repo), "--check"])
    out = tmp_path_factory.mktemp("dist") / "a.zip"
    version.main(["--root", str(repo), "--archive", str(out)])
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == [f"voicetune-{rid}/VERSION", f"voicetune-{rid}/run.sh"]
        assert z.getinfo(f"voicetune-{rid}/run.sh").external_attr >> 16 & 0o111
        assert z.read(f"voicetune-{rid}/VERSION").decode().strip() == rid


def test_repo_version_tool_runs():
    assert len(version.release_id(ROOT)) == 5
