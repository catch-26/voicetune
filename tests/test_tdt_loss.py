import math

import pytest
import torch

from tdt_loss import TDTLossTorch


def lse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def reference(acts, labels, T, L, blank, durations, sigma):
    D = len(durations)
    lp = torch.log_softmax(acts[..., :-D].double(), -1) - sigma
    dur = torch.log_softmax(acts[..., -D:].double(), -1)
    alpha = [[None] * (L + 1) for _ in range(T)]
    alpha[0][0] = 0.0
    for t in range(T):
        for u in range(L + 1):
            terms = [alpha[t][u]] if t == 0 and u == 0 else []
            for n, d in enumerate(durations):
                if d > 0 and t - d >= 0 and alpha[t - d][u] is not None:
                    terms.append(alpha[t - d][u] + lp[t - d, u, blank].item() + dur[t - d, u, n].item())
                if u > 0 and t - d >= 0 and alpha[t - d][u - 1] is not None:
                    terms.append(alpha[t - d][u - 1] + lp[t - d, u - 1, labels[u - 1]].item() + dur[t - d, u - 1, n].item())
            if not (t == 0 and u == 0):
                alpha[t][u] = lse(terms)
    ends = [alpha[T - d][L] + lp[T - d, L, blank].item() + dur[T - d, L, n].item()
            for n, d in enumerate(durations) if d > 0 and T - d >= 0 and alpha[T - d][L] is not None]
    return -lse(ends)


@pytest.mark.parametrize("durations", [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2]])
def test_matches_reference(durations):
    torch.manual_seed(0)
    B, T, U, V = 2, 6, 4, 5
    D = len(durations)
    acts = torch.randn(B, T, U, V + 1 + D)
    labels = torch.tensor([[1, 2, 3], [4, 1, 0]])
    act_lens = torch.tensor([6, 5])
    label_lens = torch.tensor([3, 2])
    loss = TDTLossTorch(blank=V, durations=durations, sigma=0.05)(acts, labels, act_lens, label_lens)
    for b in range(B):
        T_b, L_b = int(act_lens[b]), int(label_lens[b])
        ref = reference(acts[b, :T_b, :L_b + 1], labels[b].tolist(), T_b, L_b, V, durations, 0.05)
        assert loss[b].item() == pytest.approx(ref, rel=1e-4, abs=1e-4)


def test_gradients_are_finite():
    torch.manual_seed(1)
    acts = torch.randn(3, 8, 5, 7 + 1 + 5, requires_grad=True)
    labels = torch.randint(0, 7, (3, 4))
    loss = TDTLossTorch(blank=7, durations=[0, 1, 2, 3, 4], sigma=0.05)(acts, labels, torch.tensor([8, 7, 6]), torch.tensor([4, 3, 2]))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(acts.grad).all() and acts.grad.abs().sum() > 0
