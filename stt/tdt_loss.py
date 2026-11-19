import torch
NEG = -1e4

class TDTLossTorch(torch.nn.Module):
    def __init__(s, blank, durations, sigma):
        super().__init__(); s.blank, s.durations, s.sigma, s.reduction = blank, list(durations), sigma, None

    def forward(s, acts, labels, act_lens, label_lens):
        acts = acts.float(); D = len(s.durations)
        lp = torch.log_softmax(acts[..., :-D], -1) - s.sigma
        dur = torch.log_softmax(acts[..., -D:], -1)
        B, T, U, _ = lp.shape
        blank = lp[..., s.blank]
        y = labels.long().clamp_min(0)[:, None, :U - 1, None].expand(B, T, U - 1, 1)
        emit = lp[:, :, :U - 1].gather(-1, y).squeeze(-1)
        del lp
        neg = acts.new_full((B, 1), NEG)
        alpha = []
        for t in range(T):
            terms = []
            for n, d in enumerate(s.durations):
                if d == 0 or t - d < 0: continue
                a = alpha[t - d]
                terms.append(a + blank[:, t - d] + dur[:, t - d, :, n])
                terms.append(torch.cat([neg, a[:, :-1] + emit[:, t - d] + dur[:, t - d, :-1, n]], 1))
            if t == 0: c = torch.cat([acts.new_zeros(B, 1), acts.new_full((B, U - 1), NEG)], 1)
            elif terms: c = torch.logsumexp(torch.stack(terms), 0)
            else: c = acts.new_full((B, U), NEG)
            if 0 in s.durations:
                w = emit[:, t] + dur[:, t, :-1, s.durations.index(0)]
                S = torch.cat([acts.new_zeros(B, 1), w.cumsum(1)], 1)
                c = S + torch.logcumsumexp(c - S, 1)
            alpha.append(c)
        A = torch.stack(alpha, 1)
        b = torch.arange(B, device=acts.device); ll = label_lens.long(); al = act_lens.long()
        out = []
        for n, d in enumerate(s.durations):
            if d == 0: continue
            t = al - d; ok = t >= 0; t = t.clamp_min(0)
            v = A[b, t, ll] + blank[b, t, ll] + dur[b, t, ll, n]
            out.append(torch.where(ok, v, torch.full_like(v, NEG)))
        return -torch.logsumexp(torch.stack(out), 0)
