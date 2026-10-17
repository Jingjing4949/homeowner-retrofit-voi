"""H-A2 value-of-information decomposition robustness (see H_A2_PROTOCOL_20261015_CN.md).

Information split is the H-A1S S1 setting; NN, policy and metric code are the frozen H-A1 code.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import h_a0 as A
import h_a1 as H
import h_a1s  # noqa: F401  (sets the S1 information split on h_a1)

ROOT = Path(__file__).resolve().parent
ALT_DIR = A.CACHE / "emis_alt"
SCENARIOS = {"lrmer_mid_15": A.EMIS,
             "lrmer_high_re_15": "out.emissions.total.lrmer_high_re_cost_15..co2e_kg",
             "lrmer_low_re_15": "out.emissions.total.lrmer_low_re_cost_15..co2e_kg",
             "aer_mid_avg": "out.emissions.total.aer_mid_case_avg..co2e_kg"}
ALT = {k: v for k, v in SCENARIOS.items() if v != A.EMIS}
GBM_TIERS = {"BROAD": A.BROAD, "BILLS": A.BILLS, "S0": H.S0, "RICH": A.RICH}
POLICIES = ["ENTROPY", "INFO", "STATIC_GREEDY", "STATIC_GREEDY_CZ", "EVOI"]
K_LEVELS = [1, 2, 3, 4]


def fetch_alt():
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    ALT_DIR.mkdir(exist_ok=True)
    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    manifest = {}
    for k in [0] + A.ACTIONS:
        out = ALT_DIR / f"upgrade{k}.parquet"
        if not out.exists():
            t = time.time()
            tb = pq.read_table(A.S3_PREFIX + f"upgrade{k}.parquet", columns=["bldg_id", *ALT.values()],
                               filesystem=s3)
            pq.write_table(tb, out)
            print(f"upgrade{k}: {tb.num_rows} rows, {time.time() - t:.1f}s", flush=True)
        manifest[out.name] = {"sha256": A.sha256(out), "bytes": out.stat().st_size}
    (ALT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


def scenario_values(ids, db, app, col):
    """Same value definition as h_a0.load, with another emission column."""
    def read(k):
        e = pd.read_parquet(ALT_DIR / f"upgrade{k}.parquet", columns=["bldg_id", col]).set_index("bldg_id")[col]
        e = e.reindex(ids)
        assert not e.isna().any(), f"missing {col} in upgrade{k}"
        return e.to_numpy()
    e0 = read(0)
    val = np.zeros_like(db)
    for j, k in enumerate(A.ACTIONS):
        val[:, j] = np.where(app[:, j].astype(bool) & (db[:, j] >= 0), e0 - read(k), 0.0)
    return val


def cpu_count():
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, min(32, int(int(q) / int(p))))
    except (OSError, ValueError):
        pass
    return max(1, min(32, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()))


def run_scenario(name, base, val, db, role, groups, sel, eval_mask, smoke, dev, t0):
    w = base["weight"].to_numpy()
    fit = (role == "train") & ~sel
    ev = np.where(eval_mask)[0]
    vev, wev = val[ev], w[ev]
    out = {}

    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}

    n_est = 50 if smoke else A.LGB_PARAMS["n_estimators"]
    gbm_role = role.copy()
    for t, feats in GBM_TIERS.items():
        pred, iters = A.fit_predict(base, val, w, gbm_role, feats, eval_mask, n_est)
        values[f"GBM_{t}"] = A.realized(vev, A.choose(pred))
        out[f"gbm_iters_{t}"] = iters
        H.log(f"[{name}] GBM {t} V {H.wavg(values[f'GBM_{t}'], wev):.1f}", t0)

    xc, xn, sizes = H.encode(base, fit)
    scale = float(np.sqrt(np.average((val[fit] ** 2).mean(1), weights=w[fit])))
    wn = w / w[fit].mean()
    es_m = role == "es"
    tr = H.Data(xc[fit], xn[fit], val[fit] / scale, wn[fit], dev)
    es = H.Data(xc[es_m], xn[es_m], val[es_m] / scale, wn[es_m], dev)
    vnets, cnets = [], []
    for s in H.CFG["seeds"]:
        vnets.append(H.train("value", tr, es, sizes, H.CFG["epochs"], s, dev, t0)[0])
        cnets.append(H.train("cond", tr, es, sizes, H.CFG["epochs"], 100 + s, dev, t0)[0])
    del tr, es
    eng = H.Engine(vnets, cnets, sizes, scale, torch.tensor(xc, device=dev), torch.tensor(xn, device=dev), dev)
    rows = torch.tensor(ev, device=dev)
    values["NN_FULL"] = A.realized(vev, eng.decide(rows, torch.ones(len(ev), H.NQ, dtype=torch.bool,
                                                                     device=dev)).cpu().numpy())
    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()
    kmax = max(K_LEVELS)
    sel_rows = torch.tensor(np.where(sel)[0], device=dev)
    order_g = H.static_greedy(eng, sel_rows, val[sel], w[sel], kmax)
    zone_orders = {z: H.static_greedy(eng, torch.tensor(np.where(sel & (zone == z))[0], device=dev),
                                      val[sel & (zone == z)], w[sel & (zone == z)], kmax)
                   for z in np.unique(zone[sel]) if (sel & (zone == z)).sum() >= H.MIN_ZONE_ROWS}
    out["static_order"] = [H.Q[q] for q in order_g]
    uz, zi = np.unique(zone[ev], return_inverse=True)
    orders = {"STATIC_GREEDY": torch.tensor(order_g, device=dev)[None, :].expand(len(ev), -1),
              "STATIC_GREEDY_CZ": torch.tensor(np.array([zone_orders.get(z, order_g) for z in uz])[zi], device=dev)}
    gen = torch.Generator(device=dev).manual_seed(20261015)
    for p in POLICIES:
        ch, _, _ = eng.sequential(rows, p, kmax, orders=orders.get(p), gen=gen)
        for k, c in enumerate(ch):
            values[f"{p}@{k}"] = A.realized(vev, c.cpu().numpy())
        H.log(f"[{name}] {p} V {[round(H.wavg(values[f'{p}@{k}'], wev), 1) for k in range(kmax + 1)]}", t0)
    del eng, vnets, cnets
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    V = {k: H.wavg(x, wev) for k, x in values.items()}
    G = V["ORACLE"] - V["FIX_SCZ"]
    C = {k: (v - V["FIX_SCZ"]) / G for k, v in V.items()}
    boot = A.cluster_boot(groups[ev], wev, values, n_boot=H.CFG["n_boot"])
    bG = boot["ORACLE"] - boot["FIX_SCZ"]
    bC = {k: (boot[k] - boot["FIX_SCZ"]) / bG for k in values}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def stat(f):
        return {"est": float(f(C)), "ci": ci(f(bC))}

    best_static = lambda c: np.mean([np.maximum(c[f"STATIC_GREEDY@{k}"], c[f"STATIC_GREEDY_CZ@{k}"])
                                     for k in K_LEVELS], 0)
    best_info = lambda c: np.mean([np.maximum(c[f"INFO@{k}"], c[f"ENTROPY@{k}"]) for k in K_LEVELS], 0)
    evoi = lambda c: np.mean([c[f"EVOI@{k}"] for k in K_LEVELS], 0)
    d = {
        "site_gain_gbm": stat(lambda c: c["GBM_S0"] - c["GBM_BILLS"]),
        "bills_gain_gbm": stat(lambda c: c["GBM_BILLS"] - c["GBM_BROAD"]),
        "questions_gain_gbm": stat(lambda c: c["GBM_RICH"] - c["GBM_S0"]),
        "rho3": stat(lambda c: (c["EVOI@3"] - c["EVOI@0"]) / (c["NN_FULL"] - c["EVOI@0"])),
        "D_static": stat(lambda c: evoi(c) - best_static(c)),
        "D_info": stat(lambda c: evoi(c) - best_info(c)),
        "residual_gap_nn_full": stat(lambda c: 1 - c["NN_FULL"]),
        "learner_gap_full": stat(lambda c: c["GBM_RICH"] - c["NN_FULL"]),
        "learner_gap_s0": stat(lambda c: c["GBM_S0"] - c["EVOI@0"]),
    }
    hi_s = d["D_static"]["ci"][1]
    lo_s = d["D_static"]["ci"][0]
    lo_i = d["D_info"]["ci"][0]
    d["adaptive_decision"] = ("ADAPTIVE_GAIN" if lo_s >= H.MARGIN and lo_i > 0 else
                              "ADAPTIVE_GAIN_VS_STATIC_ONLY" if lo_s >= H.MARGIN else
                              "STATIC_SUFFICIENT" if hi_s < H.MARGIN else "INCONCLUSIVE")
    d["R1_site"] = d["site_gain_gbm"]["ci"][0] > 0
    d["R2_short_questionnaire"] = d["rho3"]["ci"][0] >= 0.5
    d["R3_static_sufficient"] = d["adaptive_decision"] == "STATIC_SUFFICIENT"
    out.update({"V": V, "C": C, "C_ci": {k: ci(v) for k, v in bC.items()}, "G": G, "r": G / V["ORACLE"],
                "decomposition": d})
    H.log(f"[{name}] r {G / V['ORACLE']:.3f}; C " + ", ".join(
        f"{k} {C[k]:.3f}" for k in ["GBM_BROAD", "GBM_BILLS", "GBM_S0", "GBM_RICH", "EVOI@0", "STATIC_GREEDY_CZ@3",
                                    "EVOI@3", "NN_FULL"]), t0)
    H.log(f"[{name}] " + "; ".join(f"{k} {v['est']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]"
                                    for k, v in d.items() if isinstance(v, dict)), t0)
    H.log(f"[{name}] {d['adaptive_decision']} R1 {d['R1_site']} R2 {d['R2_short_questionnaire']} "
          f"R3 {d['R3_static_sufficient']}", t0)
    return out


def run(stage):
    smoke = stage == "smoke"
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    A.LGB_PARAMS["n_jobs"] = cpu_count()
    base, val0, db, de, app, checks = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, H.SALT + "-SEL", 10) == 0)
    if smoke:
        keep = np.where((base["in.state"] == "CO").to_numpy())[0]
        base, val0, db, app, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val0[keep], db[keep],
                                                  app[keep], role[keep], groups[keep], sel[keep])
        H.CFG.update(epochs=2, seeds=(1,), n_boot=50)
        out_dir, eval_mask = ROOT / "smoke_output_a2", sel
    elif stage == "test":
        out_dir, eval_mask = ROOT / "output_a2_test", role == "test"
    else:
        out_dir, eval_mask = ROOT / "output_a2", role == "tune"
    out_dir.mkdir(exist_ok=True)
    H.log(f"device {dev}; lgb n_jobs {A.LGB_PARAMS['n_jobs']}; eval {eval_mask.sum()}", t0)
    ids = base["bldg_id"].to_numpy()
    report = {"smoke": smoke, "stage": stage, "cfg": dict(H.CFG), "lgb_n_jobs": A.LGB_PARAMS["n_jobs"], "S0": H.S0,
              "scenarios": SCENARIOS, "alt_manifest": json.loads((ALT_DIR / "MANIFEST.json").read_text()),
              "results": {}}
    for name, col in SCENARIOS.items():
        val = val0 if col == A.EMIS else scenario_values(ids, db, app, col)
        report["results"][name] = run_scenario(name, base, val, db, role, groups, sel, eval_mask, smoke, dev, t0)
        (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    res = report["results"]
    report["verdict"] = {k: all(res[s]["decomposition"][k] for s in SCENARIOS)
                         for k in ["R1_site", "R2_short_questionnaire", "R3_static_sufficient"]}
    if stage == "run":
        p = res["lrmer_mid_15"]
        report["integrity"] = {
            "V_FIX_SCZ": p["V"]["FIX_SCZ"], "V_ORACLE": p["V"]["ORACLE"],
            "C_GBM_vs_H_A0": {t: p["C"][f"GBM_{t}"] for t in ("BROAD", "BILLS", "RICH")},
            "match": (abs(p["V"]["FIX_SCZ"] - 2555.6) < 0.1 and abs(p["V"]["ORACLE"] - 3002.9) < 0.1 and
                      abs(p["C"]["GBM_BROAD"] - 0.582) < 0.01 and abs(p["C"]["GBM_BILLS"] - 0.623) < 0.01 and
                      abs(p["C"]["GBM_RICH"] - 0.908) < 0.01)}
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"verdict {report['verdict']}; integrity {report.get('integrity', {}).get('match')}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "smoke", "run", "test"])
    a = ap.parse_args()
    if a.stage == "fetch":
        fetch_alt()
    else:
        run(a.stage)
