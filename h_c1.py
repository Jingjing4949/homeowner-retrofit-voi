"""H-C1 robustness of Q1-Q3 to network retraining and to errors in homeowner answers (see H_C1_PROTOCOL_20261017_CN.md).

Main emission scenario, held-out test splits of both datasets, frozen H-A5/H-A6 definitions. Four independently
trained value-network ensembles; homeowner answers replaced at evaluation time, with probability e, by a draw from
the item's weighted training distribution. Question orders are chosen on the selection set from true answers.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import h_a0 as A
import h_a1 as H
import h_a2
import h_a5
import h_a6

ROOT = Path(__file__).resolve().parent
NH = h_a6.NH
SCEN = "lrmer_mid_15"
SEED_SETS = [(1, 2, 3), (11, 12, 13), (21, 22, 23), (31, 32, 33)]
ERRORS = [0.0, 0.1, 0.2, 0.3]
K = 3
NOISE_SEED = 20261017
PRIMARY_ERROR = 0.2


def perturb(xc, ev, fit, w, e, rng):
    out = xc.copy()
    for q in range(NH):
        j = H.J0 + q
        p = np.bincount(xc[fit, j], weights=w[fit])
        p = p / p.sum()
        hit = ev[rng.random(len(ev)) < e]
        out[hit, j] = rng.choice(len(p), size=len(hit), p=p)
    return out


def run_dataset(dataset, smoke, dev, t0):
    if dataset == "B":
        import h_b1
        h_b1.configure()
    base, val, _, _, _, _ = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, H.SALT + "-SEL", 10) == 0)
    seed_sets = SEED_SETS
    if smoke:
        keep = np.where((base["in.state"] == "CO").to_numpy())[0]
        base, val, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val[keep], role[keep],
                                        groups[keep], sel[keep])
        H.CFG.update(epochs=2, n_boot=50)
        seed_sets = [(1,), (11,)]
        eval_mask = sel
    else:
        eval_mask = role == "test"
    w = base["weight"].to_numpy()
    fit = (role == "train") & ~sel
    ev = np.where(eval_mask)[0]
    vev, wev = val[ev], w[ev]
    H.log(f"dataset {dataset}; n {len(base)}; eval {len(ev)}; actions {len(A.ACTIONS)}", t0)
    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}

    xc, xn, sizes = H.encode(base, fit)
    scale = float(np.sqrt(np.average((val[fit] ** 2).mean(1), weights=w[fit])))
    wn = w / w[fit].mean()
    es_m = role == "es"
    tr = H.Data(xc[fit], xn[fit], val[fit] / scale, wn[fit], dev)
    es = H.Data(xc[es_m], xn[es_m], val[es_m] / scale, wn[es_m], dev)
    rng = np.random.default_rng(NOISE_SEED)
    noisy = {e: perturb(xc, ev, fit, w, e, rng) for e in ERRORS}
    changed = {str(e): float(np.average((noisy[e][ev, H.J0:H.J0 + NH] != xc[ev, H.J0:H.J0 + NH]).mean(1),
                                        weights=wev)) for e in ERRORS}
    xc_true, xn_t = torch.tensor(xc, device=dev), torch.tensor(xn, device=dev)
    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()
    uz, zi = np.unique(zone[ev], return_inverse=True)
    rows = torch.tensor(ev, device=dev)
    n = len(ev)
    out = {"answers_changed_share": changed, "orders": {}, "training": {}}
    for seeds in seed_sets:
        tag = f"s{seeds[0]}"
        vnets, info = [], []
        for s in seeds:
            net, hist = H.train("value", tr, es, sizes, H.CFG["epochs"], s, dev, t0)
            vnets.append(net)
            info.append({k: hist[k] for k in ("best_es_loss", "best_epoch")})
        out["training"][tag] = info
        eng = H.Engine(vnets, [], sizes, scale, xc_true, xn_t, dev)
        order_g = h_a6.static_greedy_home(eng, torch.tensor(np.where(sel)[0], device=dev), val[sel], w[sel], K)
        zone_orders = {z: h_a6.static_greedy_home(eng, torch.tensor(np.where(sel & (zone == z))[0], device=dev),
                                                  val[sel & (zone == z)], w[sel & (zone == z)], K)
                       for z in np.unique(zone[sel]) if (sel & (zone == z)).sum() >= H.MIN_ZONE_ROWS}
        out["orders"][tag] = {"national": [H.Q[q] for q in order_g],
                              "zones": {z: [H.Q[q] for q in o] for z, o in zone_orders.items()}}
        orders = torch.tensor(np.array([zone_orders.get(z, order_g) for z in uz])[zi], device=dev)
        none = torch.zeros(n, H.NQ, dtype=torch.bool, device=dev)
        home = none.clone()
        home[:, :NH] = True
        full = torch.ones(n, H.NQ, dtype=torch.bool, device=dev)
        static = none.clone()
        static[torch.arange(n, device=dev)[:, None], orders[:, :K]] = True
        values[f"{tag}:NONE"] = A.realized(vev, eng.decide(rows, none).cpu().numpy())
        for e in ERRORS:
            eng.xc = torch.tensor(noisy[e], device=dev)
            for key, obs in (("HOME", home), ("FULL", full), ("STATIC3", static)):
                values[f"{tag}:{key}@{e}"] = A.realized(vev, eng.decide(rows, obs).cpu().numpy())
            H.log(f"[{dataset} {tag} e={e}] V none {H.wavg(values[f'{tag}:NONE'], wev):.1f} home "
                  f"{H.wavg(values[f'{tag}:HOME@{e}'], wev):.1f} static3 {H.wavg(values[f'{tag}:STATIC3@{e}'], wev):.1f}"
                  f" full {H.wavg(values[f'{tag}:FULL@{e}'], wev):.1f}", t0)
        del eng, vnets
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    del tr, es, xc_true, xn_t

    V = {k: H.wavg(x, wev) for k, x in values.items()}
    G = V["ORACLE"] - V["FIX_SCZ"]
    C = {k: (v - V["FIX_SCZ"]) / G for k, v in V.items()}
    boot = A.cluster_boot(groups[ev], wev, values, n_boot=H.CFG["n_boot"])
    bG = boot["ORACLE"] - boot["FIX_SCZ"]
    bC = {k: (boot[k] - boot["FIX_SCZ"]) / bG for k in values}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def stat(f):
        return {"est": float(f(C)), "ci": ci(f(bC))}

    res = {}
    for seeds in seed_sets:
        tag = f"s{seeds[0]}"
        for e in ERRORS:
            g = lambda c, e=e, tag=tag: c[f"{tag}:HOME@{e}"] - c[f"{tag}:NONE"]
            d = {"nn_home_gain": stat(g),
                 "nn_audit_gain": stat(lambda c: c[f"{tag}:FULL@{e}"] - c[f"{tag}:HOME@{e}"]),
                 "home_minus_audit": stat(lambda c: 2 * c[f"{tag}:HOME@{e}"] - c[f"{tag}:NONE"] - c[f"{tag}:FULL@{e}"]),
                 "rho3_home_static": stat(lambda c: (c[f"{tag}:STATIC3@{e}"] - c[f"{tag}:NONE"]) / g(c)),
                 "home_gain_retained": stat(lambda c: g(c) / (c[f"{tag}:HOME@0.0"] - c[f"{tag}:NONE"]))}
            d["Q1_home_value"] = d["nn_home_gain"]["ci"][0] > 0
            d["Q2_home_over_audit"] = d["home_minus_audit"]["ci"][0] > 0
            d["Q3_short_home_questionnaire"] = d["rho3_home_static"]["ci"][0] >= 0.5
            res[f"{tag}@{e}"] = d
            H.log(f"[{dataset} {tag} e={e}] " + "; ".join(
                f"{k} {v['est']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]" if isinstance(v, dict) else f"{k} {v}"
                for k, v in d.items()), t0)
    claims = h_a6.CLAIMS
    out.update(V=V, G=G, C=C, results=res, verdict={
        "C1_retraining": all(res[f"s{s[0]}@0.0"][k] for s in seed_sets for k in claims),
        "C2_answer_error": all(res[f"s{s[0]}@{PRIMARY_ERROR}"][k] for s in seed_sets for k in claims)})
    if not smoke:
        ref = h_a6.REF[dataset][SCEN]
        out["integrity"] = {"FIX_SCZ": V["FIX_SCZ"], "ORACLE": V["ORACLE"],
                            "match": abs(V["FIX_SCZ"] - ref[0]) < 0.1 and abs(V["ORACLE"] - ref[1]) < 0.1}
    return out


def run(smoke):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    out_dir = ROOT / ("smoke_output_c1" if smoke else "output_c1")
    out_dir.mkdir(exist_ok=True)
    report = {"smoke": smoke, "cfg": dict(H.CFG), "seed_sets": SEED_SETS, "errors": ERRORS, "k": K,
              "noise_seed": NOISE_SEED, "primary_error": PRIMARY_ERROR, "home": h_a5.HOME, "results": {}}
    for dataset in ("A", "B"):
        try:
            report["results"][dataset] = run_dataset(dataset, smoke, dev, t0)
        except ZeroDivisionError:
            if not smoke:
                raise
            H.log(f"[{dataset}] smoke: 2-epoch network is constant, ratio undefined", t0)
        (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"verdict { {d: r['verdict'] for d, r in report['results'].items()} }; integrity "
          f"{ {d: r.get('integrity', {}).get('match') for d, r in report['results'].items()} }", t0)


if __name__ == "__main__":
    assert sys.argv[1] in ("smoke", "run")
    run(sys.argv[1] == "smoke")
