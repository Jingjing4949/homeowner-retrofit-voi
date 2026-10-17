"""H-A5 four-tier value-of-information decomposition on the tune set (see H_A5_PROTOCOL_20261016_CN.md).

Tiers: records (BILLS) -> homeowner questionnaire -> professional audit -> thermal response test (ground conductivity).
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import h_a0 as A
import h_a1 as H
import h_a1s
import h_a2

ROOT = Path(__file__).resolve().parent
GTC = "in.ground_thermal_conductivity"
AUDIT = sorted(h_a1s.INSPECT)
HOME = [c for c in A.RICH if c not in A.BILLS and c != GTC and c not in h_a1s.INSPECT]
TIERS = {"T1_RECORDS": A.BILLS, "T2_HOME": A.BILLS + HOME, "T3_AUDIT": A.BILLS + HOME + AUDIT, "T4_TEST": A.RICH}
TEST_FRACTIONS = [0.1, 0.2, 0.3, 0.5]
K_LEVELS = [1, 2, 3, 4]
POLICIES = ["ENTROPY", "INFO", "STATIC_GREEDY", "STATIC_GREEDY_CZ", "EVOI"]

H.S0 = A.BILLS
H.Q = HOME + AUDIT
H.S0_CAT = [c for c in H.S0 if c not in A.NUMERIC]
H.S0_NUM = [c for c in H.S0 if c in A.NUMERIC]
H.NQ = len(H.Q)
H.J0 = len(H.S0_CAT)
assert len(HOME) == 15 and len(AUDIT) == 10 and GTC not in H.Q


def lgb_fit(X, y, w, Xes, yes, wes, n_est, cls=False):
    p = {**A.LGB_PARAMS, "n_estimators": n_est}
    if cls:
        p.pop("objective")
        m = lgb.LGBMClassifier(**p)
    else:
        m = lgb.LGBMRegressor(**p)
    m.fit(X, y, sample_weight=w, eval_set=[(Xes, yes)], eval_sample_weight=[wes],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def with_gtc(X, level, cats):
    Z = X.copy()
    Z[GTC] = pd.Categorical([level] * len(Z), categories=cats)
    return Z


def plus(f):
    """Value of the best action, 0 if no retrofit is better."""
    return np.maximum(f.max(-1), 0)


def act(f):
    return np.where(f.max(-1) > 0, f.argmax(-1) + 1, 0)


def test_targeting(name, base, val, w, role, ev, n_est, t0):
    """Thermal-response-test targeting with the T4 value models and a T3 classifier for conductivity."""
    tr, es = role == "train", role == "es"
    cats = base[GTC].cat.categories
    X4 = base[A.RICH]
    Xev = X4.iloc[ev]
    f_true = np.zeros((len(ev), val.shape[1]))
    f_lvl = np.zeros((len(cats), len(ev), val.shape[1]))
    lvl_X = [with_gtc(Xev, c, cats) for c in cats]
    for j in range(val.shape[1]):
        m = lgb_fit(X4[tr], val[tr, j], w[tr], X4[es], val[es, j], w[es], n_est)
        f_true[:, j] = m.predict(Xev)
        for i, Z in enumerate(lvl_X):
            f_lvl[i, :, j] = m.predict(Z)
    X3 = base[TIERS["T3_AUDIT"]]
    y = base[GTC].cat.codes.to_numpy()
    clf = lgb_fit(X3[tr], y[tr], w[tr], X3[es], y[es], w[es], n_est, cls=True)
    p = np.zeros((len(ev), len(cats)))
    p[:, clf.classes_] = clf.predict_proba(X3.iloc[ev])
    f_exp = np.einsum("ni,inj->nj", p, f_lvl)
    evoi = (p * plus(f_lvl).T).sum(1) - plus(f_exp)
    choice_untested, choice_tested = act(f_exp), act(f_true)
    log(f"[{name}] targeting models done; conductivity classifier accuracy "
        f"{np.average(p.argmax(1) == y[ev], weights=w[ev]):.3f}", t0)
    return evoi, choice_untested, choice_tested, p


def rank_mask(score, w, q):
    o = np.argsort(-score, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    m = np.zeros(len(score), bool)
    m[o[cw <= q + 1e-12]] = True
    return m


def run_scenario(name, base, val, role, groups, sel, eval_mask, smoke, dev, t0):
    w = base["weight"].to_numpy()
    fit = (role == "train") & ~sel
    ev = np.where(eval_mask)[0]
    vev, wev = val[ev], w[ev]
    n_est = 50 if smoke else A.LGB_PARAMS["n_estimators"]
    out = {}
    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}
    t3_choice = None
    for t, feats in TIERS.items():
        if t == "T4_TEST":
            continue
        pred, _ = A.fit_predict(base, val, w, role, feats, eval_mask, n_est)
        if t == "T3_AUDIT":
            t3_choice = A.choose(pred)
        values[f"GBM_{t}"] = A.realized(vev, A.choose(pred))
        log(f"[{name}] GBM {t} V {H.wavg(values[f'GBM_{t}'], wev):.1f}", t0)

    evoi, ch_u, ch_t, p = test_targeting(name, base, val, w, role, ev, n_est, t0)
    values["GBM_T4_TEST"] = A.realized(vev, ch_t)
    values["T3_EXPECTED"] = A.realized(vev, ch_u)
    untested = values["GBM_T3_AUDIT"]
    gshp = np.isin(t3_choice - 1, [A.ACTIONS.index(k) for k in (6, 7, 8, 18, 31, 32)]) & (t3_choice > 0)
    rng = np.random.default_rng(20261016)
    rand_score = rng.random(len(ev))
    q_gshp = float(np.average(gshp, weights=wev))
    for q in TEST_FRACTIONS + [round(q_gshp, 4)]:
        for kind, score in (("EVOI", evoi), ("RANDOM", rand_score)):
            m = rank_mask(score, wev, q)
            values[f"TEST_{kind}@{q}"] = np.where(m, values["GBM_T4_TEST"], untested)
    values[f"TEST_GSHP_RULE@{round(q_gshp, 4)}"] = np.where(gshp, values["GBM_T4_TEST"], untested)
    o = np.argsort(-evoi)
    cw = np.cumsum(wev[o]) / wev.sum()
    out["evoi_profile"] = {"share_positive": float(np.average(evoi > 1e-9, weights=wev)),
                           "mean": float(np.average(evoi, weights=wev)),
                           "top_share_of_total": {str(q): float((evoi[o] * wev[o])[cw <= q].sum() / (evoi * wev).sum())
                                                  for q in TEST_FRACTIONS},
                           "gshp_rule_test_fraction": q_gshp}

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
    gen = torch.Generator(device=dev).manual_seed(20261016)
    asked = {}
    for pol in POLICIES:
        ch, ak, _ = eng.sequential(rows, pol, kmax, orders=orders.get(pol), gen=gen)
        for k, c in enumerate(ch):
            values[f"{pol}@{k}"] = A.realized(vev, c.cpu().numpy())
        asked[pol] = ak.cpu().numpy()
        log(f"[{name}] {pol} V {[round(H.wavg(values[f'{pol}@{k}'], wev), 1) for k in range(kmax + 1)]}", t0)
    out["question_share_evoi"] = {H.Q[q]: float(np.average((asked["EVOI"] == q).any(1), weights=wev))
                                  for q in range(H.NQ)}
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
    evoi_k = lambda c: np.mean([c[f"EVOI@{k}"] for k in K_LEVELS], 0)
    phi = lambda key: (lambda c: (c[key] - c["GBM_T3_AUDIT"]) / (c["GBM_T4_TEST"] - c["GBM_T3_AUDIT"]))
    d = {
        "home_gain": stat(lambda c: c["GBM_T2_HOME"] - c["GBM_T1_RECORDS"]),
        "audit_gain": stat(lambda c: c["GBM_T3_AUDIT"] - c["GBM_T2_HOME"]),
        "test_gain": stat(lambda c: c["GBM_T4_TEST"] - c["GBM_T3_AUDIT"]),
        "residual_gap_t4": stat(lambda c: 1 - c["GBM_T4_TEST"]),
        "rho3": stat(lambda c: (c["EVOI@3"] - c["EVOI@0"]) / (c["NN_FULL"] - c["EVOI@0"])),
        "D_static": stat(lambda c: evoi_k(c) - best_static(c)),
        "D_info": stat(lambda c: evoi_k(c) - best_info(c)),
        "learner_gap_t3": stat(lambda c: c["GBM_T3_AUDIT"] - c["NN_FULL"]),
    }
    for q in TEST_FRACTIONS:
        d[f"phi_evoi@{q}"] = stat(phi(f"TEST_EVOI@{q}"))
        d[f"phi_evoi_minus_random@{q}"] = stat(lambda c, q=q: phi(f"TEST_EVOI@{q}")(c) - phi(f"TEST_RANDOM@{q}")(c))
    qg = round(q_gshp, 4)
    d["phi_gshp_rule"] = stat(phi(f"TEST_GSHP_RULE@{qg}"))
    d["phi_evoi_minus_gshp_rule_same_fraction"] = stat(
        lambda c: phi(f"TEST_EVOI@{qg}")(c) - phi(f"TEST_GSHP_RULE@{qg}")(c))
    lo_s, hi_s, lo_i = d["D_static"]["ci"][0], d["D_static"]["ci"][1], d["D_info"]["ci"][0]
    d["adaptive_decision"] = ("ADAPTIVE_GAIN" if lo_s >= H.MARGIN and lo_i > 0 else
                              "ADAPTIVE_GAIN_VS_STATIC_ONLY" if lo_s >= H.MARGIN else
                              "STATIC_SUFFICIENT" if hi_s < H.MARGIN else "INCONCLUSIVE")
    d["R1_home"] = d["home_gain"]["ci"][0] > 0
    d["R2_short_questionnaire"] = d["rho3"]["ci"][0] >= 0.5
    d["R3_static_sufficient"] = d["adaptive_decision"] == "STATIC_SUFFICIENT"
    d["R4_targeted_test"] = d["phi_evoi@0.2"]["ci"][0] >= 0.5 and d["phi_evoi_minus_random@0.2"]["ci"][0] > 0
    out.update({"V": V, "C": C, "C_ci": {k: ci(v) for k, v in bC.items()}, "G": G, "r": G / V["ORACLE"],
                "decomposition": d})
    H.log(f"[{name}] r {G / V['ORACLE']:.3f}; C " + ", ".join(
        f"{k} {C[k]:.3f}" for k in ["GBM_T1_RECORDS", "GBM_T2_HOME", "GBM_T3_AUDIT", "GBM_T4_TEST", "T3_EXPECTED",
                                    "EVOI@0", "STATIC_GREEDY_CZ@3", "EVOI@3", "NN_FULL"]), t0)
    H.log(f"[{name}] " + "; ".join(f"{k} {v['est']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]"
                                    for k, v in d.items() if isinstance(v, dict)), t0)
    H.log(f"[{name}] {d['adaptive_decision']} R1 {d['R1_home']} R2 {d['R2_short_questionnaire']} "
          f"R3 {d['R3_static_sufficient']} R4 {d['R4_targeted_test']}", t0)
    return out


def log(msg, t0):
    H.log(msg, t0)


def run(smoke):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    A.LGB_PARAMS["n_jobs"] = h_a2.cpu_count()
    base, val0, db, de, app, checks = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, H.SALT + "-SEL", 10) == 0)
    if smoke:
        keep = np.where((base["in.state"] == "CO").to_numpy())[0]
        base, val0, db, app, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val0[keep], db[keep],
                                                  app[keep], role[keep], groups[keep], sel[keep])
        H.CFG.update(epochs=2, seeds=(1,), n_boot=50)
        out_dir, eval_mask = ROOT / "smoke_output_a5", sel
    else:
        out_dir, eval_mask = ROOT / "output_a5", role == "tune"
    out_dir.mkdir(exist_ok=True)
    H.log(f"device {dev}; lgb n_jobs {A.LGB_PARAMS['n_jobs']}; Q {H.NQ}; eval {eval_mask.sum()}", t0)
    ids = base["bldg_id"].to_numpy()
    report = {"smoke": smoke, "cfg": dict(H.CFG), "tiers": TIERS, "scenarios": h_a2.SCENARIOS, "results": {}}
    for name, col in h_a2.SCENARIOS.items():
        val = val0 if col == A.EMIS else h_a2.scenario_values(ids, db, app, col)
        report["results"][name] = run_scenario(name, base, val, role, groups, sel, eval_mask, smoke, dev, t0)
        (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    res = report["results"]
    report["verdict"] = {k: all(res[s]["decomposition"][k] for s in h_a2.SCENARIOS)
                         for k in ["R1_home", "R2_short_questionnaire", "R3_static_sufficient", "R4_targeted_test"]}
    if not smoke:
        p = res["lrmer_mid_15"]
        report["integrity"] = {
            "V_FIX_SCZ": p["V"]["FIX_SCZ"], "V_ORACLE": p["V"]["ORACLE"],
            "C": {k: p["C"][k] for k in ("GBM_T1_RECORDS", "GBM_T3_AUDIT", "GBM_T4_TEST")},
            "match": (abs(p["V"]["FIX_SCZ"] - 2555.6) < 0.1 and abs(p["V"]["ORACLE"] - 3002.9) < 0.1 and
                      abs(p["C"]["GBM_T1_RECORDS"] - 0.623) < 0.01 and abs(p["C"]["GBM_T3_AUDIT"] - 0.739) < 0.01 and
                      abs(p["C"]["GBM_T4_TEST"] - 0.908) < 0.01)}
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"verdict {report['verdict']}; integrity {report.get('integrity', {}).get('match')}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "run"])
    run(smoke=ap.parse_args().stage == "smoke")
