"""H-A1S question-pool realism sensitivity (see H_A1S_PROTOCOL_20261015_CN.md).

Reuses the frozen H-A1 model, policy and metric code; only the information split and question costs change.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import h_a0 as A
import h_a1 as H

ROOT = Path(__file__).resolve().parent
SITE = ["in.ground_thermal_conductivity"]
INSPECT = {"in.hvac_heating_efficiency", "in.hvac_cooling_efficiency", "in.duct_leakage_and_insulation",
           "in.insulation_wall", "in.insulation_ceiling", "in.insulation_floor", "in.insulation_foundation_wall",
           "in.insulation_roof", "in.insulation_slab", "in.infiltration"}
INSPECT_COST = 2
BUDGETS = [2, 3, 4, 6]
ADAPTIVE = ["RANDOM", "ENTROPY", "INFO", "EVOI"]

H.S0 = A.BILLS + SITE
H.Q = [c for c in A.RICH if c not in H.S0]
H.S0_CAT = [c for c in H.S0 if c not in A.NUMERIC]
H.S0_NUM = [c for c in H.S0 if c in A.NUMERIC]
H.NQ = len(H.Q)
H.J0 = len(H.S0_CAT)
COST = np.array([INSPECT_COST if q in INSPECT else 1 for q in H.Q], dtype=float)
assert H.NQ == 25 and len(INSPECT & set(H.Q)) == len(INSPECT)


def value_of(eng, rows, val, w, obs):
    return H.wavg(A.realized(val, eng.decide(rows, obs).cpu().numpy()), w)


def static_cost(eng, rows, val, w, budget):
    """Greedy by value gain per cost on the selection rows; stops when no affordable question gains value."""
    chosen, spent = [], 0.0
    obs = torch.zeros(len(rows), H.NQ, dtype=torch.bool, device=eng.dev)
    cur = value_of(eng, rows, val, w, obs)
    while True:
        best_q, best_r, best_v = None, 0.0, None
        for q in range(H.NQ):
            if q in chosen or COST[q] > budget - spent:
                continue
            o = obs.clone()
            o[:, q] = True
            v = value_of(eng, rows, val, w, o)
            if (v - cur) / COST[q] > best_r:
                best_q, best_r, best_v = q, (v - cur) / COST[q], v
        if best_q is None:
            return chosen
        chosen.append(best_q)
        obs[:, best_q] = True
        spent += COST[best_q]
        cur = best_v


def adaptive_cost(eng, rows, kind, budget, gen):
    n = len(rows)
    cost = torch.tensor(COST, device=eng.dev, dtype=torch.float32)
    obs = torch.zeros(n, H.NQ, dtype=torch.bool, device=eng.dev)
    left = torch.full((n,), float(budget), device=eng.dev)
    asked = torch.zeros(n, device=eng.dev)
    for _ in range(int(budget)):
        afford = (~obs) & (cost[None, :] <= left[:, None])
        active = afford.any(1)
        if not active.any():
            break
        if kind == "RANDOM":
            s = torch.rand(n, H.NQ, generator=gen, device=eng.dev)
        else:
            s, _ = eng.adaptive_scores(rows, obs, kind)
            s = s / cost[None, :]
        q = s.masked_fill(~afford, float("-inf")).argmax(1)
        idx = torch.where(active)[0]
        obs[idx, q[idx]] = True
        left[idx] -= cost[q[idx]]
        asked[idx] += 1
    return eng.decide(rows, obs), asked


def run(smoke):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    base, val, db, de, app, checks = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, H.SALT + "-SEL", 10) == 0)
    if smoke:
        keep = np.where((base["in.state"] == "CO").to_numpy())[0]
        base, val, db, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val[keep], db[keep],
                                            role[keep], groups[keep], sel[keep])
        H.CFG.update(epochs=2, seeds=(1,), n_boot=50)
        kmax, out_dir, eval_role, budgets = 2, ROOT / "smoke_output_s", "sel", [2]
    else:
        kmax, out_dir, eval_role, budgets = H.KMAX, ROOT / "output_s", "tune", BUDGETS
    out_dir.mkdir(exist_ok=True)
    w = base["weight"].to_numpy()
    fit = (role == "train") & ~sel
    ev_mask = sel if eval_role == "sel" else role == "tune"
    H.log(f"device {dev}; Q={H.NQ}, S0 cat {len(H.S0_CAT)}; eval {ev_mask.sum()} ({eval_role})", t0)

    xc, xn, sizes = H.encode(base, fit)
    scale = float(np.sqrt(np.average((val[fit] ** 2).mean(1), weights=w[fit])))
    wn = w / w[fit].mean()
    es_m = role == "es"
    tr = H.Data(xc[fit], xn[fit], val[fit] / scale, wn[fit], dev)
    es = H.Data(xc[es_m], xn[es_m], val[es_m] / scale, wn[es_m], dev)
    report = {"smoke": smoke, "cfg": dict(H.CFG), "Q": H.Q, "S0": H.S0, "cost": dict(zip(H.Q, COST.tolist())),
              "training": {}}
    vnets, cnets = [], []
    for s in H.CFG["seeds"]:
        m, h = H.train("value", tr, es, sizes, H.CFG["epochs"], s, dev, t0)
        vnets.append(m)
        report["training"][f"value_{s}"] = h
        m, h = H.train("cond", tr, es, sizes, H.CFG["epochs"], 100 + s, dev, t0)
        cnets.append(m)
        report["training"][f"cond_{s}"] = h
    del tr, es
    eng = H.Engine(vnets, cnets, sizes, scale, torch.tensor(xc, device=dev), torch.tensor(xn, device=dev), dev)

    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()
    sel_rows = torch.tensor(np.where(sel)[0], device=dev)
    zone_sel = {z: sel & (zone == z) for z in np.unique(zone[sel]) if (sel & (zone == z)).sum() >= H.MIN_ZONE_ROWS}
    ev = np.where(ev_mask)[0]
    rows = torch.tensor(ev, device=dev)
    vev, wev, dbe = val[ev], w[ev], db[ev]
    uz, zi = np.unique(zone[ev], return_inverse=True)

    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}
    choices = {"FIX_SCZ": fix}
    full = torch.ones(len(ev), H.NQ, dtype=torch.bool, device=dev)
    choices["NN_FULL"] = eng.decide(rows, full).cpu().numpy()
    values["NN_FULL"] = A.realized(vev, choices["NN_FULL"])

    # S1: unit cost, K questions (identical procedure to H-A1)
    order_g = H.static_greedy(eng, sel_rows, val[sel], w[sel], kmax)
    zone_orders = {z: H.static_greedy(eng, torch.tensor(np.where(m)[0], device=dev), val[m], w[m], kmax)
                   for z, m in zone_sel.items()}
    report["S1_static_order"] = [H.Q[q] for q in order_g]
    orders_g = torch.tensor(order_g, device=dev)[None, :].expand(len(ev), -1)
    orders_cz = torch.tensor(np.array([zone_orders.get(z, order_g) for z in uz])[zi], device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261015)
    asked = {}
    for p in H.POLICIES:
        ch, ak, _ = eng.sequential(rows, p, kmax, orders={"STATIC_GREEDY": orders_g,
                                                          "STATIC_GREEDY_CZ": orders_cz}.get(p), gen=gen)
        for k, c in enumerate(ch):
            choices[f"S1:{p}@{k}"] = c.cpu().numpy()
            values[f"S1:{p}@{k}"] = A.realized(vev, choices[f"S1:{p}@{k}"])
        asked[p] = ak.cpu().numpy()
        H.log(f"S1 {p}: V {[round(H.wavg(values[f'S1:{p}@{k}'], wev), 1) for k in range(kmax + 1)]}", t0)

    # S2: inspection questions cost 2, budget B in cost units
    report["S2_static_sets"] = {}
    n_asked = {}
    for B in budgets:
        set_g = static_cost(eng, sel_rows, val[sel], w[sel], B)
        set_z = {z: static_cost(eng, torch.tensor(np.where(m)[0], device=dev), val[m], w[m], B)
                 for z, m in zone_sel.items()}
        report["S2_static_sets"][str(B)] = {"national": [H.Q[q] for q in set_g],
                                            "zones": {z: [H.Q[q] for q in s] for z, s in set_z.items()}}
        obs = torch.zeros(len(ev), H.NQ, dtype=torch.bool, device=dev)
        obs[:, set_g] = True
        choices[f"S2:STATIC_GREEDY@{B}"] = eng.decide(rows, obs).cpu().numpy()
        obs = torch.zeros(len(ev), H.NQ, dtype=torch.bool, device=dev)
        for i, z in enumerate(uz):
            obs[torch.tensor(np.where(zi == i)[0], device=dev)[:, None],
                torch.tensor(set_z.get(z, set_g), device=dev, dtype=torch.long)[None, :]] = True
        choices[f"S2:STATIC_GREEDY_CZ@{B}"] = eng.decide(rows, obs).cpu().numpy()
        for p in ADAPTIVE:
            ch, na = adaptive_cost(eng, rows, p, B, gen)
            choices[f"S2:{p}@{B}"] = ch.cpu().numpy()
            n_asked[f"S2:{p}@{B}"] = float(np.average(na.cpu().numpy(), weights=wev))
        for p in ["STATIC_GREEDY", "STATIC_GREEDY_CZ"] + ADAPTIVE:
            values[f"S2:{p}@{B}"] = A.realized(vev, choices[f"S2:{p}@{B}"])
        H.log(f"S2 B={B}: " + ", ".join(f"{p} {H.wavg(values[f'S2:{p}@{B}'], wev):.1f}"
                                         for p in ["STATIC_GREEDY", "STATIC_GREEDY_CZ"] + ADAPTIVE), t0)

    V = {k: H.wavg(x, wev) for k, x in values.items()}
    G = V["ORACLE"] - V["FIX_SCZ"]
    C = {k: (v - V["FIX_SCZ"]) / G for k, v in V.items()}
    boot = A.cluster_boot(groups[ev], wev, values, n_boot=H.CFG["n_boot"])
    bG = boot["ORACLE"] - boot["FIX_SCZ"]
    bC = {k: (boot[k] - boot["FIX_SCZ"]) / bG for k in values}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def endpoint(prefix, levels):
        bs = np.mean([np.maximum(bC[f"{prefix}:STATIC_GREEDY@{k}"], bC[f"{prefix}:STATIC_GREEDY_CZ@{k}"])
                      for k in levels], 0)
        bi = np.mean([np.maximum(bC[f"{prefix}:INFO@{k}"], bC[f"{prefix}:ENTROPY@{k}"]) for k in levels], 0)
        be = np.mean([bC[f"{prefix}:EVOI@{k}"] for k in levels], 0)
        ps = np.mean([max(C[f"{prefix}:STATIC_GREEDY@{k}"], C[f"{prefix}:STATIC_GREEDY_CZ@{k}"]) for k in levels])
        pi = np.mean([max(C[f"{prefix}:INFO@{k}"], C[f"{prefix}:ENTROPY@{k}"]) for k in levels])
        pe = np.mean([C[f"{prefix}:EVOI@{k}"] for k in levels])
        out = {"levels": levels, "D_static": float(pe - ps), "D_static_ci": ci(be - bs),
               "D_info": float(pe - pi), "D_info_ci": ci(be - bi)}
        lo_s, hi_s, lo_i = out["D_static_ci"][0], out["D_static_ci"][1], out["D_info_ci"][0]
        out["decision"] = ("ADAPTIVE_GAIN" if lo_s >= H.MARGIN and lo_i > 0 else
                           "ADAPTIVE_GAIN_VS_STATIC_ONLY" if lo_s >= H.MARGIN else
                           "STATIC_SUFFICIENT" if hi_s < H.MARGIN else "INCONCLUSIVE")
        return out

    res = {"V": V, "C": C, "C_ci": {k: ci(v) for k, v in bC.items()}, "G": G, "r": G / V["ORACLE"],
           "nn_full_recovery": C["NN_FULL"],
           "S1": endpoint("S1", [k for k in H.PRIMARY_K if k <= kmax]),
           "S2": endpoint("S2", budgets), "S2_mean_questions_asked": n_asked}
    res["S1_question_share"] = {p: {H.Q[q]: float(np.average((asked[p] == q).any(1), weights=wev))
                                    for q in range(H.NQ)} for p in H.POLICIES}
    res["bill_increase_rate"] = {}
    for k, c in choices.items():
        inc = np.zeros(len(c), bool)
        r_ = c > 0
        inc[r_] = dbe[np.where(r_)[0], c[r_] - 1] < 0
        res["bill_increase_rate"][k] = float(np.average(inc, weights=wev))
    if not smoke:
        report["integrity"] = {"V_FIX_SCZ": V["FIX_SCZ"], "V_ORACLE": V["ORACLE"],
                               "match": abs(V["FIX_SCZ"] - 2555.6) < 0.1 and abs(V["ORACLE"] - 3002.9) < 0.1}
    report["result"] = res
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"NN_FULL C {C['NN_FULL']:.3f}", t0)
    for p in H.POLICIES:
        H.log(f"S1 C {p}: {[round(C[f'S1:{p}@{k}'], 3) for k in range(kmax + 1)]}", t0)
    for B in budgets:
        H.log(f"S2 C B={B}: " + ", ".join(f"{p} {C[f'S2:{p}@{B}']:.3f}"
                                         for p in ["STATIC_GREEDY", "STATIC_GREEDY_CZ"] + ADAPTIVE), t0)
    for s in ("S1", "S2"):
        o = res[s]
        H.log(f"{s}: D_static {o['D_static']:.4f} {o['D_static_ci']}  D_info {o['D_info']:.4f} "
              f"{o['D_info_ci']}  decision {o['decision']}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "run"])
    run(smoke=ap.parse_args().stage == "smoke")
