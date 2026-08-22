"""SPEAR-LLM: char-level Transformer on tiny_shakespeare with algebraic SPEAR activations.

Subcommands:
  fetch                        download tiny_shakespeare (HF, fallback GitHub)
  train --act NAME [--steps N] train and save out/<NAME>.pt
  sample --ckpt PATH [--n N]   generate text
  kv                           KV-cache eviction experiment on out/*.pt
"""
import argparse
import json
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "tiny_shakespeare.txt")
CONSTS_PATH = os.path.join(HERE, "spear_constants.json")
RESULTS = os.path.join(OUT, "results.json")

PAPER = {"silu": [0.5, 0.516445, 0.01, 2.853773],
         "gelu": [0.492929, 0.275808, 0.005871, 1.000622],
         "gelu2": [0.5, 0.50559, 0.001, 0.682199],
         "phi": [1.061244, 0.009764, 3.856829],
         "softplus": [9.172138, 9.182172, 5.48952]}


def consts():
    d = dict(PAPER)
    if os.path.exists(CONSTS_PATH):
        with open(CONSTS_PATH) as f:
            d.update(json.load(f))
    return d


def make_acts(C):
    a, b, c, d = C["silu"]
    ga, gb, gc, gk = C["gelu"]
    sa, sb, sc = C["softplus"]

    def spear_silu(x):
        return x * (a + b * x / (c + torch.sqrt(d + x * x)))

    def spear_gelu(x):
        return x * torch.clamp(torch.clamp((gc * x + gb) * x + ga, min=0.0), max=gk)

    ga2, gb2, gc2, gd2 = C["gelu2"]

    def spear_gelu2(x):
        return x * (ga2 + gb2 * x / (gc2 + torch.sqrt(gd2 + x * x)))

    def spear_softplus(x):
        u = torch.exp(-torch.abs(x))
        return torch.clamp(x, min=0.0) + u * (sa + u) / (sb + sc * u)

    def spear_sigmoid(x):
        return 1.0 - 1.0 / (1.0 + torch.exp(-x))

    return {"silu": F.silu, "gelu": F.gelu, "spear_silu": spear_silu,
            "spear_gelu": spear_gelu, "spear_gelu2": spear_gelu2,
            "spear_softplus": spear_softplus, "spear_sigmoid": spear_sigmoid}


# ---------------- model ----------------
class SoftmaxAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d, self.h = d, h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.use_sdpa = False

    def forward(self, x, atts=None, knorms=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        hd = self.d // self.h
        q = q.view(B, T, self.h, hd).transpose(1, 2)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)
        if getattr(self, "use_sdpa", False) and atts is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            if atts is not None:
                atts.append(att.detach())
                knorms.append(k.norm(dim=-1).mean(1).detach())
            y = att @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, self.d))


class LinearAttn(nn.Module):
    """BT13 Cauchy-Schwarz linear attention, O(N) causal, no exp / no softmax.
    O_i = (Σ_{j<=i} k_j ⊗ v_j) · q_i / (q_i · Σ_{j<=i} k_j)
    Prefix sums via cumsum -> true O(N) in sequence length (constant hd² per head).
    normalize=True: unit-norm q,k -> scale-invariant cosine (Cauchy-Schwarz).
    """

    def __init__(self, d, h, normalize=True):
        super().__init__()
        self.d, self.h = d, h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.normalize = normalize
        self.eps = 1e-6

    def forward(self, x, atts=None, knorms=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        hd = self.d // self.h
        q = q.view(B, T, self.h, hd).transpose(1, 2)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)
        if self.normalize:
            q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
            k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)          # (B,h,T,hd,hd)
        kv_cum = torch.cumsum(outer, dim=2)                # O(N) causal prefix
        k_cum = torch.cumsum(k, dim=2)
        o = (kv_cum @ q.unsqueeze(-1)).squeeze(-1)         # (B,h,T,hd)
        den = (q * k_cum).sum(dim=-1, keepdim=True)
        o = o / (den + self.eps)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, self.d))


class DecayedLinearAttn(nn.Module):
    """Mur A : RetNet-style decayed linear attention — recency via décroissance
    exponentielle apprise par tête, toujours O(N) par cumsum re-scalé :
        S_t = lam·S_{t-1} + k_t ⊗ v_t   =>   S_t = lam^t · cumsum(k⊗v / lam^j)
    lam par tête borné [0.50,0.95] (sigmoid param) : sûr en float32 pour T<=~300.
    """

    def __init__(self, d, h, lam_lo=0.60, lam_hi=0.95):
        super().__init__()
        self.d, self.h = d, h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.eps = 1e-6
        self.lam_lo, self.lam_hi = lam_lo, lam_hi
        init = torch.linspace(lam_lo + 0.08, lam_hi - 0.02, h)
        u = (init - lam_lo) / (lam_hi - lam_lo)
        self.lam_logit = nn.Parameter(torch.log(u / (1 - u)))

    def forward(self, x, atts=None, knorms=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        hd = self.d // self.h
        q = q.view(B, T, self.h, hd).transpose(1, 2)       # (B,h,T,hd)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)
        # interne float64 : lam^-j atteint ~1e20+, float32 deborde en backward
        dt = torch.float64
        lam = (torch.sigmoid(self.lam_logit.double()) * (self.lam_hi - self.lam_lo)
               + self.lam_lo)                              # (h,)
        pos = torch.arange(T, device=x.device, dtype=dt)
        lpow = lam[:, None] ** pos[None, :]                # (h,T) = lam^t
        inv = 1.0 / lpow                                   # lam^-j
        qd, kd, vd = q.to(dt), k.to(dt), v.to(dt)
        ks = kd * inv.unsqueeze(0).unsqueeze(-1)
        outer = ks.unsqueeze(-1) * vd.unsqueeze(-2)        # (B,h,T,hd,hd)
        kv = torch.cumsum(outer, dim=2) * lpow.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        kc = torch.cumsum(ks, dim=2) * lpow.unsqueeze(0).unsqueeze(-1)
        o = (kv @ qd.unsqueeze(-1)).squeeze(-1)            # (B,h,T,hd)
        den = (qd * kc).sum(dim=-1, keepdim=True)
        o = o / (den + self.eps)
        o = o.to(v.dtype)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, self.d))


class Block(nn.Module):
    def __init__(self, d, h, act, kind="softmax"):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if kind == "softmax":
            self.attn = SoftmaxAttn(d, h)
        elif kind == "linearraw":
            self.attn = LinearAttn(d, h, normalize=False)
        elif kind == "decayed":
            self.attn = DecayedLinearAttn(d, h)
        else:
            self.attn = LinearAttn(d, h, normalize=True)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.act = act

    def forward(self, x, atts=None, knorms=None):
        x = x + self.attn(self.ln1(x), atts, knorms)
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))


class TernaryLinear(nn.Linear):
    """Mur B : poids ternaires {-g,0,+g} (absmean BitNet-style) entraînés via STE.
    Forward quantifié, backward identité -> le réseau APPREND dans le régime ternaire.
    Mémoire packée théorique : 2 bits/poids vs 32."""

    def forward(self, x):
        w = self.weight
        gamma = w.abs().mean().detach().clamp_min(1e-8)
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0) * gamma
        return F.linear(x, w_q + (w - w.detach()), self.bias)


def to_ternary_(mod, skip=("head",)):
    """Remplace récursivement les nn.Linear (sauf skip) par des TernaryLinear."""
    for name, child in mod.named_children():
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            new = TernaryLinear(child.in_features, child.out_features,
                                bias=child.bias is not None)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(mod, name, new)
        else:
            to_ternary_(child, skip)


def packed_bytes_kb(m):
    """Taille packée : 2 bits/poids pour les TernaryLinear, fp32 pour le reste."""
    seen, lin_bytes = set(), 0.0
    for mod in m.modules():
        if isinstance(mod, TernaryLinear):
            p = mod.weight.data_ptr()
            if p not in seen:
                seen.add(p)
                lin_bytes += mod.weight.numel() * 0.25
    other = 0.0
    for _, t in m.state_dict().items():
        p = t.data_ptr()
        if p not in seen:
            seen.add(p)
            other += t.numel() * t.element_size()
    return (lin_bytes + other) / 1024.0


class GPT(nn.Module):
    def __init__(self, vocab, d=128, nl=4, h=4, T=192, act=F.silu, attn="softmax"):
        super().__init__()
        self.T = T
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(T, d)
        if attn == "hybrid":
            kinds = ["softmax" if i % 2 == 0 else "linear" for i in range(nl)]
        else:
            kinds = [attn] * nl
        self.blocks = nn.ModuleList([Block(d, h, act, k) for k in kinds])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, atts=None, knorms=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x, atts, knorms)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------- data ----------------
def fetch(_):
    urls = ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/tiny_shakespeare.txt"]
    for u in urls:
        try:
            urllib.request.urlretrieve(u, DATA)
            print("downloaded:", u)
            return
        except Exception as e:
            print("fail:", u, e)
    raise SystemExit("no source reachable")


def load_data():
    if not os.path.exists(DATA):
        fetch(None)
    text = open(DATA, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = np.array(chars)
    arr = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(arr))
    return torch.from_numpy(arr[:n]), torch.from_numpy(arr[n:]), len(chars), itos


# ---------------- train / sample / kv ----------------
def get_batch(src, B, T):
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i + T] for i in ix])
    y = torch.stack([src[i + 1:i + T + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate(m, va, B, T, n=10):
    m.eval()
    ls = [m(*get_batch(va, B, T))[1].item() for _ in range(n)]
    m.train()
    return float(np.mean(ls))


def cmd_train(a):
    tr, va, V, itos = load_data()
    act = make_acts(consts())[a.act]
    d, nl, h, T, B = a.d, a.nl, 4, 128, 16
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    m = GPT(V, d, nl, h, T, act, attn=a.attn)
    if not getattr(a, "nosdpa", False):
        for blk in m.blocks:
            if hasattr(blk.attn, "use_sdpa"):
                blk.attn.use_sdpa = True
    if getattr(a, "ternary", False):
        to_ternary_(m)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    warm = max(10, a.steps // 20)

    def lr_at(s):
        if s < warm:
            return 3e-3 * (s + 1) / warm
        p = (s - warm) / max(1, a.steps - warm)
        return 3e-4 + 0.5 * (3e-3 - 3e-4) * (1.0 + math.cos(math.pi * p))

    t0 = time.time()
    print(f"act={a.act} params={nparam/1e6:.2f}M steps={a.steps} B={B} T={T}")
    for s in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = get_batch(tr, B, T)
        _, loss = m(x, y)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            print(f"step {s+1}: loss non-fini -> step ignore", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (s + 1) % 100 == 0 or s == 0 or s == a.steps - 1:
            vl = estimate(m, va, B, T)
            print(f"step {s+1:4d} loss {loss.item():.4f} val {vl:.4f} "
                  f"({(s+1)/(time.time()-t0):.1f} it/s)", flush=True)
    vl = estimate(m, va, B, T)
    sec = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    tag = a.act if a.attn == "softmax" and a.seed == 1337 else f"{a.act}_{a.attn}_s{a.seed}"
    if getattr(a, "ternary", False):
        tag += "_tern"
    if a.d != 96 or a.nl != 3:
        tag += f"_d{a.d}n{a.nl}"
    if getattr(a, "ternary", False):
        print(f"packed memory (2-bit ternary linears): {packed_bytes_kb(m):.1f} KB")
    torch.save({"model": m.state_dict(), "cfg": dict(vocab=V, d=d, nl=nl, h=h, T=T),
                "act": a.act, "attn": a.attn, "seed": a.seed, "val": vl,
                "steps": a.steps, "sec": sec, "params": nparam},
               os.path.join(OUT, tag + ".pt"))
    res = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            res = json.load(f)
    res = [r for r in res if not (r["act"] == a.act and r.get("attn", "softmax") == a.attn
                                  and r.get("seed", 1337) == a.seed and r.get("steps") == a.steps
                                  and r.get("d", 96) == a.d and r.get("nl", 3) == a.nl)]
    res.append(dict(act=a.act, attn=a.attn, seed=a.seed, steps=a.steps, d=a.d, nl=a.nl,
                    val_loss=vl, ppl=math.exp(vl), sec=sec, params=nparam,
                    ips=a.steps / sec))
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    print(f"DONE act={a.act} attn={a.attn} seed={a.seed} val={vl:.4f} ppl={math.exp(vl):.2f} time={sec:.0f}s")


def _remap(sd):
    """Old ckpts stored qkv/proj directly on Block -> nest under .attn."""
    out = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) > 3 and parts[0] == "blocks" and parts[2] in ("qkv", "proj"):
            parts.insert(2, "attn")
            k = ".".join(parts)
        out[k] = v
    return out


def load_ckpt(path):
    ck = torch.load(path, weights_only=False)
    cfg = ck["cfg"]
    act = make_acts(consts())[ck["act"]]
    m = GPT(cfg["vocab"], cfg["d"], cfg["nl"], cfg["h"], cfg["T"], act,
            attn=cfg.get("attn", "softmax"))
    m.load_state_dict(_remap(ck["model"]))
    m.eval()
    return m, cfg


def cmd_sample(a):
    _, _, _, itos = load_data()
    m, cfg = load_ckpt(a.ckpt)
    idx = torch.tensor([[0]])  # '\n'
    with torch.no_grad():
        for _ in range(a.n):
            logits, _ = m(idx[:, -cfg["T"]:])
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:, -1, :], -1), 1)], 1)
    print("".join(itos[i] for i in idx[0].tolist()))


@torch.no_grad()
def cmd_kv(_):
    _, va, _, _ = load_data()
    paths = [os.path.join(OUT, f"{n}.pt") for n in
             ["silu", "spear_silu", "gelu", "spear_gelu", "spear_gelu2"]]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no checkpoints in out/ — run train first")
    B, frac = 16, 0.5
    print(f"{'ckpt':14s} {'random':>7s} {'window':>7s} {'S_only':>7s} {'A(H2O)':>7s} {'4S+A+1.5R':>10s} {'(A+R)(1+3S)':>12s} {'jac(rule,H2O)':>14s}")
    for p in paths:
        m, cfg = load_ckpt(p)
        if cfg.get("attn", "softmax") != "softmax":
            print(f"{os.path.basename(p):14s}  (linear attn: no explicit attention weights — skipped)")
            continue
        T, P = cfg["T"], int(0.75 * cfg["T"])
        x, _ = get_batch(va, B, T)
        atts, knorms = [], []
        m(x, atts=atts, knorms=knorms)
        A = torch.stack(atts).mean((0, 2))[:, P:, :P].sum(1)   # (B,P) future attention mass
        S = torch.stack(knorms).mean(0)[:, :P]                 # (B,P) key-norm salience
        R = torch.linspace(0, 1, P).unsqueeze(0).expand(B, P)

        def norm(z):
            return (z - z.min(1, keepdim=True).values) / (z.max(1, keepdim=True).values - z.min(1, keepdim=True).values + 1e-9)

        An, Sn = norm(A), norm(S)
        rule = 4 * Sn + An + 1.5 * R            # additive triad
        rule_m = (An + R) * (1.0 + 3.0 * Sn)    # multiplicative triad (SPEAR codex)
        k = P // 2

        def ret(score):
            keep = torch.zeros_like(A, dtype=torch.bool).scatter_(
                1, torch.topk(score, k, dim=1).indices, True)
            return ((A * keep).sum(1) / A.sum(1)).mean().item()

        keep_h2o = torch.zeros_like(A, dtype=torch.bool).scatter_(
            1, torch.topk(An, k, dim=1).indices, True)
        keep_rule = torch.zeros_like(A, dtype=torch.bool).scatter_(
            1, torch.topk(rule, k, dim=1).indices, True)
        jac = (keep_h2o & keep_rule).sum(1).float() / (keep_h2o | keep_rule).sum(1)
        win = ((A[:, -k:]).sum(1) / A.sum(1)).mean().item()
        print(f"{os.path.basename(p):14s} {ret(torch.rand(B, P)):7.3f} {win:7.3f} "
              f"{ret(Sn):7.3f} {ret(An):7.3f} {ret(rule):10.3f} {ret(rule_m):12.3f} {jac.mean().item():14.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    t = sub.add_parser("train")
    t.add_argument("--act", required=True,
                   choices=["silu", "gelu", "spear_silu", "spear_gelu", "spear_gelu2"])
    t.add_argument("--attn", default="softmax",
                   choices=["softmax", "linear", "linearraw", "hybrid", "decayed"])
    t.add_argument("--ternary", action="store_true")
    t.add_argument("--no-sdpa", action="store_true", dest="nosdpa",
                   help="attention manuelle au lieu du kernel fusé SDPA")
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--steps", type=int, default=500)
    t.add_argument("--d", type=int, default=96)
    t.add_argument("--nl", type=int, default=3)
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--n", type=int, default=400)
    sub.add_parser("kv")
    a = p.parse_args()
    dict(fetch=fetch, train=cmd_train, sample=cmd_sample, kv=cmd_kv)[a.cmd](a)


if __name__ == "__main__":
    main()
