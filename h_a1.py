"""H-A1 adaptive questioning test on ResStock 2025 R1 (see H_A1_PROTOCOL_20261014_CN.md)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import h_a0 as A

ROOT = Path(__file__).resolve().parent
SALT = "HA1-20261014"
S0 = A.BILLS
Q = [c for c in A.RICH if c not in A.BILLS]
S0_CAT = [c for c in S0 if c not in A.NUMERIC]
S0_NUM = [c for c in S0 if c in A.NUMERIC]
NQ = len(Q)
J0 = len(S0_CAT)
KMAX = 6
PRIMARY_K = [1, 2, 3, 4]
TOPM = 6
MIN_ZONE_ROWS = 500
MARGIN = 0.02
CFG = dict(emb=16, hidden=(512, 512, 256), dropout=0.1, lr=1e-3, wd=1e-5, batch=2048, epochs=30,
           seeds=(1, 2, 3), n_boot=1000)
POLICIES = ["RANDOM", "ENTROPY", "INFO", "STATIC_GREEDY", "STATIC_GREEDY_CZ", "EVOI"]


def log(msg, t0):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def encode(base, fit):
    cats = S0_CAT + Q
    X = np.zeros((len(base), len(cats)), np.int64)
    sizes = []
    for j, c in enumerate(cats):
        s = base[c].astype(str).to_numpy()
        idx = {v: i + 2 for i, v in enumerate(sorted(set(s[fit])))}  # 0 = masked, 1 = unseen
        X[:, j] = [idx.get(v, 1) for v in s]
        sizes.append(len(idx) + 2)
    num = np.log1p(np.clip(base[S0_NUM].to_numpy(float), 0, None))
    mu, sd = np.nanmean(num[fit], 0), np.nanstd(num[fit], 0) + 1e-6
    N = np.nan_to_num((num - mu) / sd).astype(np.float32)
    return X, N, sizes


class Net(nn.Module):
    def __init__(self, sizes, n_num, out):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(s, CFG["emb"]) for s in sizes)
        d, layers = CFG["emb"] * len(sizes) + n_num, []
        for h in CFG["hidden"]:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(CFG["dropout"])]
            d = h
        self.body, self.head = nn.Sequential(*layers), nn.Linear(d, out)

    def forward(self, xc, xn):
        z = torch.cat([e(xc[:, j]) for j, e in enumerate(self.emb)] + [xn], 1)
        return self.head(self.body(z))


def masked(xc, obs):
    out = xc.clone()
    out[:, J0:] = torch.where(obs, out[:, J0:], torch.zeros_like(out[:, J0:]))
    return out


def rand_obs(n, gen, dev, hi):
    k = torch.randint(0, hi + 1, (n, 1), generator=gen, device=dev)
    rank = torch.rand(n, NQ, generator=gen, device=dev).argsort(1).argsort(1)
    return rank < k


class Data:
    def __init__(self, xc, xn, y, w, dev):
        self.xc, self.xn = torch.tensor(xc, device=dev), torch.tensor(xn, device=dev)
        self.y, self.w = torch.tensor(y, dtype=torch.float32, device=dev), torch.tensor(w, dtype=torch.float32, device=dev)


def cond_loss(logits_all, offsets, sizes_q, target, obs, w):
    loss, cnt = 0.0, 0.0
    for q in range(NQ):
        lg = logits_all[:, offsets[q]:offsets[q] + sizes_q[q]]
        m = (~obs[:, q]).float() * w
        ce = nn.functional.cross_entropy(lg, target[:, q], reduction="none")
        loss = loss + (ce * m).sum()
        cnt = cnt + m.sum()
    return loss / cnt.clamp_min(1.0)


def train(kind, tr, es, sizes, epochs, seed, dev, t0):
    torch.manual_seed(seed)
    sizes_q = sizes[J0:]
    offsets = np.concatenate([[0], np.cumsum(sizes_q)])
    out = tr.y.shape[1] if kind == "value" else int(offsets[-1])
    net = Net(sizes, tr.xn.shape[1], out).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    gen = torch.Generator(device=dev).manual_seed(seed)
    hi = NQ if kind == "value" else NQ - 1
    es_gen = torch.Generator(device=dev).manual_seed(7)
    es_obs = rand_obs(len(es.xc), es_gen, dev, hi)

    def loss_of(d, rows, obs):
        x = masked(d.xc[rows], obs)
        p = net(x, d.xn[rows])
        if kind == "value":
            return (d.w[rows] * ((p - d.y[rows]) ** 2).mean(1)).sum() / d.w[rows].sum()
        return cond_loss(p, offsets, sizes_q, d.xc[rows][:, J0:], obs, d.w[rows])

    best, best_state, hist = float("inf"), None, []
    n = len(tr.xc)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=gen, device=dev)
        for b in range(0, n, CFG["batch"]):
            rows = perm[b:b + CFG["batch"]]
            loss = loss_of(tr, rows, rand_obs(len(rows), gen, dev, hi))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for b in range(0, len(es.xc), 65536):
                rows = torch.arange(b, min(b + 65536, len(es.xc)), device=dev)
                tot += float(loss_of(es, rows, es_obs[rows])) * len(rows)
                cnt += len(rows)
        el = tot / cnt
        hist.append(el)
        if el < best:
            best, best_state = el, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    log(f"{kind} seed {seed}: best es loss {best:.5f} at epoch {int(np.argmin(hist)) + 1}/{epochs}", t0)
    return net, {"best_es_loss": best, "best_epoch": int(np.argmin(hist)) + 1, "history": hist}


class Engine:
    def __init__(self, vnets, cnets, sizes, scale, xc, xn, dev):
        self.v, self.c, self.scale, self.dev = vnets, cnets, scale, dev
        self.xc, self.xn = xc, xn
        sq = sizes[J0:]
        self.off = np.concatenate([[0], np.cumsum(sq)])
        self.sq = sq

    @torch.no_grad()
    def value(self, xc, xn, bs=262144):
        outs = []
        for b in range(0, len(xc), bs):
            outs.append(torch.stack([m(xc[b:b + bs], xn[b:b + bs]) for m in self.v]).mean(0))
        return torch.cat(outs) * self.scale

    @torch.no_grad()
    def cond(self, xc, xn):
        probs = []
        raw = [m(xc, xn) for m in self.c]
        for q in range(NQ):
            sl = slice(int(self.off[q]), int(self.off[q + 1]))
            p = torch.stack([torch.softmax(r[:, sl].clone().index_fill_(1, torch.tensor([0, 1], device=self.dev),
                                                                         float("-inf")), 1) for r in raw]).mean(0)
            probs.append(p)
        return probs

    @staticmethod
    def choose(f):
        best = f.argmax(1)
        return torch.where(f.gather(1, best[:, None])[:, 0] > 0, best + 1, torch.zeros_like(best))

    def decide(self, rows, obs, bs=65536):
        ch = []
        for b in range(0, len(rows), bs):
            r = rows[b:b + bs]
            ch.append(self.choose(self.value(masked(self.xc[r], obs[b:b + bs]), self.xn[r])))
        return torch.cat(ch)

    @torch.no_grad()
    def adaptive_scores(self, rows, obs, kind, bs=4096):
        scores, mass = [], []
        for b in range(0, len(rows), bs):
            r, o = rows[b:b + bs], obs[b:b + bs]
            x0, n0 = masked(self.xc[r], o), self.xn[r]
            if kind == "ENTROPY":
                probs = self.cond(x0, n0)
                s = torch.stack([-(p.clamp_min(1e-12).log() * p).sum(1) for p in probs], 1)
            else:
                probs = self.cond(x0, n0)
                f0 = self.value(x0, n0)
                cur = self.choose(f0)
                s = torch.zeros(len(r), NQ, device=self.dev)
                cov = torch.zeros(len(r), NQ, device=self.dev)
                for q in range(NQ):
                    k = min(TOPM, int(self.sq[q]) - 2)
                    if k <= 0:
                        continue
                    pv, iv = probs[q].topk(k, 1)
                    cov[:, q] = pv.sum(1)
                    pv = pv / pv.sum(1, keepdim=True).clamp_min(1e-12)
                    xr = x0.repeat_interleave(k, 0)
                    xr[:, J0 + q] = iv.reshape(-1)
                    f = self.value(xr, n0.repeat_interleave(k, 0)).reshape(len(r), k, -1)
                    if kind == "EVOI":
                        g = f.max(2).values.clamp_min(0)
                        fa = torch.where(cur[:, None] > 0,
                                         f.gather(2, (cur.clamp_min(1) - 1)[:, None, None].expand(-1, k, 1))[..., 0],
                                         torch.zeros_like(g))
                        s[:, q] = (pv * (g - fa)).sum(1)
                    else:
                        s[:, q] = (pv * (f - f0[:, None, :]).norm(dim=2)).sum(1)
                mass.append((cov * (~o).float()).sum(1) / (~o).float().sum(1).clamp_min(1))
            scores.append(s)
        return torch.cat(scores), (torch.cat(mass) if mass else None)

    def sequential(self, rows, kind, kmax, orders=None, gen=None):
        n = len(rows)
        obs = torch.zeros(n, NQ, dtype=torch.bool, device=self.dev)
        choices, asked, cover = [self.decide(rows, obs)], [], []
        for k in range(kmax):
            if kind in ("STATIC_GREEDY", "STATIC_GREEDY_CZ"):
                q = orders[:, k]
            elif kind == "RANDOM":
                s = torch.rand(n, NQ, generator=gen, device=self.dev)
                q = s.masked_fill(obs, float("-inf")).argmax(1)
            else:
                s, m = self.adaptive_scores(rows, obs, kind)
                q = s.masked_fill(obs, float("-inf")).argmax(1)
                if m is not None:
                    cover.append(float(m.mean()))
            obs[torch.arange(n, device=self.dev), q] = True
            asked.append(q)
            choices.append(self.decide(rows, obs))
        return choices, torch.stack(asked, 1), cover


def wavg(x, w):
    return float(np.average(x, weights=w))


def static_greedy(eng, rows, val, w, kmax):
    order = []
    for _ in range(kmax):
        best_q, best_v = None, -np.inf
        for q in range(NQ):
            if q in order:
                continue
            obs = torch.zeros(len(rows), NQ, dtype=torch.bool, device=eng.dev)
            obs[:, order + [q]] = True
            v = wavg(A.realized(val, eng.decide(rows, obs).cpu().numpy()), w)
            if v > best_v:
                best_q, best_v = q, v
        order.append(best_q)
    return order


def run(smoke):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    base, val, db, de, app, checks = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, SALT + "-SEL", 10) == 0)
    if smoke:
        co = (base["in.state"] == "CO").to_numpy()
        keep = np.where(co)[0]
        base, val, db, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val[keep], db[keep],
                                            role[keep], groups[keep], sel[keep])
        CFG.update(epochs=2, seeds=(1,), n_boot=50)
        kmax, out_dir, eval_role = 2, ROOT / "smoke_output", "sel"
    else:
        kmax, out_dir, eval_role = KMAX, ROOT / "output", "tune"
    out_dir.mkdir(exist_ok=True)
    w = base["weight"].to_numpy()
    fit = (role == "train") & ~sel
    ev_mask = sel if eval_role == "sel" else role == "tune"
    log(f"device {dev}; rows fit {fit.sum()} es {(role == 'es').sum()} sel {sel.sum()} eval {ev_mask.sum()} ({eval_role})", t0)

    xc, xn, sizes = encode(base, fit)
    scale = float(np.sqrt(np.average((val[fit] ** 2).mean(1), weights=w[fit])))
    wn = w / w[fit].mean()
    tr = Data(xc[fit], xn[fit], val[fit] / scale, wn[fit], dev)
    es_m = role == "es"
    es = Data(xc[es_m], xn[es_m], val[es_m] / scale, wn[es_m], dev)
    report = {"smoke": smoke, "cfg": CFG, "Q": Q, "S0": S0, "checks": checks, "scale": scale,
              "rows": {"fit": int(fit.sum()), "es": int(es_m.sum()), "sel": int(sel.sum()), "eval": int(ev_mask.sum())},
              "training": {}}
    vnets, cnets = [], []
    for s in CFG["seeds"]:
        m, h = train("value", tr, es, sizes, CFG["epochs"], s, dev, t0)
        vnets.append(m)
        report["training"][f"value_{s}"] = h
        m, h = train("cond", tr, es, sizes, CFG["epochs"], 100 + s, dev, t0)
        cnets.append(m)
        report["training"][f"cond_{s}"] = h
    del tr, es
    eng = Engine(vnets, cnets, sizes, scale, torch.tensor(xc, device=dev), torch.tensor(xn, device=dev), dev)

    sel_rows = torch.tensor(np.where(sel)[0], device=dev)
    order_g = static_greedy(eng, sel_rows, val[sel], w[sel], kmax)
    log(f"static greedy order: {[Q[q] for q in order_g]}", t0)
    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()
    zone_orders = {}
    for z in np.unique(zone[sel]):
        zm = sel & (zone == z)
        if zm.sum() >= MIN_ZONE_ROWS:
            zone_orders[z] = static_greedy(eng, torch.tensor(np.where(zm)[0], device=dev), val[zm], w[zm], kmax)
    log(f"zone orders fitted for {len(zone_orders)} zones", t0)
    report["static_order"] = [Q[q] for q in order_g]
    report["zone_orders"] = {z: [Q[q] for q in o] for z, o in zone_orders.items()}

    ev = np.where(ev_mask)[0]
    rows = torch.tensor(ev, device=dev)
    vev, wev, dbe = val[ev], w[ev], db[ev]
    og = torch.tensor(order_g, device=dev)
    orders_g = og[None, :].expand(len(ev), -1)
    uz, zi = np.unique(zone[ev], return_inverse=True)
    table = np.array([zone_orders.get(z, order_g) for z in uz])
    orders_cz = torch.tensor(table[zi], device=dev)

    # reference policies identical to H-A0
    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}
    choices = {"FIX_SCZ": fix}
    full = torch.ones(len(ev), NQ, dtype=torch.bool, device=dev)
    choices["NN_FULL"] = eng.decide(rows, full).cpu().numpy()
    values["NN_FULL"] = A.realized(vev, choices["NN_FULL"])

    asked, cover = {}, {}
    gen = torch.Generator(device=dev).manual_seed(20261014)
    for p in POLICIES:
        ch, ak, cv = eng.sequential(rows, p, kmax,
                                    orders={"STATIC_GREEDY": orders_g, "STATIC_GREEDY_CZ": orders_cz}.get(p), gen=gen)
        for k, c in enumerate(ch):
            choices[f"{p}@{k}"] = c.cpu().numpy()
            values[f"{p}@{k}"] = A.realized(vev, choices[f"{p}@{k}"])
        asked[p] = ak.cpu().numpy()
        cover[p] = cv
        log(f"{p} done; V@0..{kmax} = {[round(wavg(values[f'{p}@{k}'], wev), 1) for k in range(kmax + 1)]}", t0)

    V = {k: wavg(x, wev) for k, x in values.items()}
    if not smoke:
        report["integrity"] = {"V_FIX_SCZ": V["FIX_SCZ"], "V_ORACLE": V["ORACLE"], "ha0_FIX_SCZ": 2555.6,
                               "ha0_ORACLE": 3002.9,
                               "match": abs(V["FIX_SCZ"] - 2555.6) < 0.1 and abs(V["ORACLE"] - 3002.9) < 0.1}
    G = V["ORACLE"] - V["FIX_SCZ"]
    C = {k: (v - V["FIX_SCZ"]) / G for k, v in V.items()}
    boot = A.cluster_boot(groups[ev], wev, values, n_boot=CFG["n_boot"])
    bG = boot["ORACLE"] - boot["FIX_SCZ"]
    bC = {k: (boot[k] - boot["FIX_SCZ"]) / bG for k in values}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    ks = [k for k in PRIMARY_K if k <= kmax]
    bstat = np.mean([np.maximum(bC[f"STATIC_GREEDY@{k}"], bC[f"STATIC_GREEDY_CZ@{k}"]) for k in ks], 0)
    binfo = np.mean([np.maximum(bC[f"INFO@{k}"], bC[f"ENTROPY@{k}"]) for k in ks], 0)
    bevoi = np.mean([bC[f"EVOI@{k}"] for k in ks], 0)
    pt = lambda p: float(np.mean([C[f"{p}@{k}"] for k in ks]))
    D = pt("EVOI") - float(np.mean([max(C[f"STATIC_GREEDY@{k}"], C[f"STATIC_GREEDY_CZ@{k}"]) for k in ks]))
    DI = pt("EVOI") - float(np.mean([max(C[f"INFO@{k}"], C[f"ENTROPY@{k}"]) for k in ks]))
    res = {"V": V, "G": G, "r": G / V["ORACLE"], "C": C, "G_ci": ci(bG), "C_ci": {k: ci(v) for k, v in bC.items()},
           "D_static": D, "D_static_ci": ci(bevoi - bstat), "D_info": DI, "D_info_ci": ci(bevoi - binfo),
           "primary_K": ks}
    lo_s, hi_s, lo_i = res["D_static_ci"][0], res["D_static_ci"][1], res["D_info_ci"][0]
    if lo_s >= MARGIN and lo_i > 0:
        decision = "ADAPTIVE_GAIN"
    elif lo_s >= MARGIN:
        decision = "ADAPTIVE_GAIN_VS_STATIC_ONLY"
    elif hi_s < MARGIN:
        decision = "STATIC_SUFFICIENT"
    else:
        decision = "INCONCLUSIVE"
    res["decision"] = decision
    res["nn_full_recovery"] = C["NN_FULL"]
    res["nn_degraded"] = C["NN_FULL"] < 0.85
    for p in POLICIES:
        for target in (0.6, 0.7, 0.8):
            hit = [k for k in range(kmax + 1) if C[f"{p}@{k}"] >= target]
            res.setdefault("k_to_reach", {}).setdefault(p, {})[str(target)] = hit[0] if hit else None
    res["bill_increase_rate"] = {}
    for k, c in choices.items():
        inc = np.zeros(len(c), bool)
        r_ = c > 0
        inc[r_] = dbe[np.where(r_)[0], c[r_] - 1] < 0
        res["bill_increase_rate"][k] = float(np.average(inc, weights=wev))
    res["question_share"] = {p: {Q[q]: float(np.average((asked[p] == q).any(1), weights=wev)) for q in range(NQ)}
                             for p in POLICIES}
    res["candidate_mass_covered"] = cover
    report["result"] = res
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"G {G:.1f} r {res['r']:.4f}; NN_FULL C {C['NN_FULL']:.3f}", t0)
    for p in POLICIES:
        log(f"C {p}: {[round(C[f'{p}@{k}'], 3) for k in range(kmax + 1)]}", t0)
    log(f"D_static {D:.4f} {res['D_static_ci']}  D_info {DI:.4f} {res['D_info_ci']}  decision {decision}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "run"])
    run(smoke=ap.parse_args().stage == "smoke")
