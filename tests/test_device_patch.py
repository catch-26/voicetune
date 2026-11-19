import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts"))
from device_patch import patch  # noqa: E402

UPSTREAM = {
    "train_first.py": '    device = accelerator.device\n                mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")\n                    wav.append(torch.from_numpy(y).to("cuda"))\n',
    "train_second.py": '    device = "cuda"\n    torch.cuda.empty_cache()\n                        mask = length_to_mask(mel_input_length // (2**n_down)).to(\n                            "cuda"\n                        )\n',
    "Modules/hifigan.py": '                        torch.ones(1, 1, F0_down).to("cuda"),\n',
    "Modules/istftnet.py": '                        torch.ones(1, 1, N_down).to("cuda"),\n',
}


def tree(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)


def test_patch_removes_hard_coded_cuda(tmp_path):
    tree(tmp_path, UPSTREAM)
    patch(str(tmp_path))
    first = (tmp_path / "train_first.py").read_text()
    second = (tmp_path / "train_second.py").read_text()
    assert first.count(".to(device)") == 2 and '"cuda"' not in first
    assert 'device = "cuda" if torch.cuda.is_available() else "cpu"' in second and ".to(device)" in second
    assert "torch.cuda.empty_cache()" in second
    for name in ("Modules/hifigan.py", "Modules/istftnet.py"):
        assert ".to(asr.device)" in (tmp_path / name).read_text()


def test_patch_is_idempotent(tmp_path):
    tree(tmp_path, UPSTREAM)
    patch(str(tmp_path))
    once = {n: (tmp_path / n).read_text() for n in UPSTREAM}
    patch(str(tmp_path))
    assert once == {n: (tmp_path / n).read_text() for n in UPSTREAM}


def test_patch_fails_on_unknown_cuda_use(tmp_path):
    tree(tmp_path, {**UPSTREAM, "Modules/hifigan.py": 'torch.ones(1).to(device="cuda")\n'})
    with pytest.raises(RuntimeError, match="hard-coded cuda"):
        patch(str(tmp_path))
