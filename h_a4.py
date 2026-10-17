"""H-A4 diagnostic: is the ground-conductivity gain specific to geothermal heat pump actions? Tune set only.

See H_A4_PROTOCOL_20261015_CN.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import h_a0 as A
import h_a2

ROOT = Path(__file__).resolve().parent
GTC = "in.ground_thermal_conductivity"
GSHP = [6, 7, 8, 18, 31, 32]
TIERS = {"BROAD": A.BROAD, "BILLS": A.BILLS, "S0": A.BILLS + [GTC], "RICH_NO_GTC": [c for c in A.RICH if c != GTC],
         "RICH": A.RICH}


def log(msg, t0):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def evaluate(name, base, val, w, role, groups, ev_mask, n_est, t0, gshp_cols):
    ev = np.where(ev_mask)[0]
    vev, wev = val[ev], w[ev]
    fix = A.fixed_policy(base, val, w, role == "train", [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                        ["in.ashrae_iecc_climate_zone_2004"]])[ev]
    values = {"FIX_SCZ": A.realized(vev, fix), "ORACLE": np.maximum(vev.max(1), 0)}
    choices = {"FIX_SCZ": fix}
    for t, feats in TIERS.items():
        pred, _ = A.fit_predict(base, val, w, role, feats, ev_mask, n_est)
        choices[t] = A.choose(pred)
        values[t] = A.realized(vev, choices[t])
        log(f"[{name}] {t} V {np.average(values[t], weights=wev):.1f}", t0)
    V = {k: float(np.average(x, weights=wev)) for k, x in values.items()}
    G = V["ORACLE"] - V["FIX_SCZ"]
    C = {k: (v - V["FIX_SCZ"]) / G for k, v in V.items()}
    boot = A.cluster_boot(groups[ev], wev, values)
    bG = boot["ORACLE"] - boot["FIX_SCZ"]
    bC = {k: (boot[k] - boot["FIX_SCZ"]) / bG for k in values}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {"V": V, "G": G, "C": C, "C_ci": {k: ci(v) for k, v in bC.items()},
           "gtc_gain_over_bills": {"est": C["S0"] - C["BILLS"], "ci": ci(bC["S0"] - bC["BILLS"]),
                                   "kg": V["S0"] - V["BILLS"], "kg_ci": ci(boot["S0"] - boot["BILLS"])},
           "gtc_gain_over_rich_no_gtc": {"est": C["RICH"] - C["RICH_NO_GTC"],
                                         "ci": ci(bC["RICH"] - bC["RICH_NO_GTC"]),
                                         "kg": V["RICH"] - V["RICH_NO_GTC"],
                                         "kg_ci": ci(boot["RICH"] - boot["RICH_NO_GTC"])}}
    labels = ["none"] + [str(a) for a in gshp_cols[1]]
    out["choice_share"] = {k: {labels[i]: float(np.average(c == i, weights=wev)) for i in np.unique(c)}
                           for k, c in choices.items()}
    oracle = np.where(vev.max(1) > 0, np.argmax(vev, 1) + 1, 0)
    out["choice_share"]["ORACLE"] = {labels[i]: float(np.average(oracle == i, weights=wev)) for i in np.unique(oracle)}
    if gshp_cols[0]:
        is_g = lambda c: np.isin(c - 1, gshp_cols[0]) & (c > 0)
        diff = (values["S0"] - values["BILLS"]) * wev
        touch = is_g(choices["S0"]) | is_g(choices["BILLS"])
        out["gtc_gain_share_gshp_households"] = float(diff[touch].sum() / diff.sum())
        out["gshp_share"] = {k: float(np.average(is_g(c), weights=wev)) for k, c in choices.items()}
        out["gshp_share"]["ORACLE"] = float(np.average(is_g(oracle), weights=wev))
    log(f"[{name}] r {G / V['ORACLE']:.3f}; C " + ", ".join(f"{k} {C[k]:.3f}" for k in TIERS) +
        f"; gtc gain {out['gtc_gain_over_bills']['est']:.4f} {out['gtc_gain_over_bills']['ci']}" +
        f"; gtc gain on RICH {out['gtc_gain_over_rich_no_gtc']['est']:.4f} {out['gtc_gain_over_rich_no_gtc']['ci']}", t0)
    return out


def run(smoke):
    t0 = time.time()
    A.LGB_PARAMS["n_jobs"] = h_a2.cpu_count()
    base, val, db, de, app, checks = A.load()
    role, groups = A.split(base)
    if smoke:
        keep = (base["in.state"] == "CO").to_numpy()
        base, val, role, groups = base[keep].reset_index(drop=True), val[keep], role[keep], groups[keep]
        sub = A.bucket(groups, A.SALT + "-SMOKE", 2)
        role = np.where(role == "train", np.where(sub == 0, "train", "es"), "excluded")
        ev_mask, n_est, out_dir = role == "es", 50, ROOT / "smoke_output_a4"
    else:
        ev_mask, n_est, out_dir = role == "tune", A.LGB_PARAMS["n_estimators"], ROOT / "output_a4"
    out_dir.mkdir(exist_ok=True)
    w = base["weight"].to_numpy()
    log(f"lgb n_jobs {A.LGB_PARAMS['n_jobs']}; eval {ev_mask.sum()}", t0)

    gtc = base[GTC].astype(str)
    x = gtc.astype(float)
    report = {"smoke": smoke, "gshp_actions": GSHP, "gtc_distribution": gtc.value_counts(normalize=True).to_dict(),
              "gtc_r2_by": {}}
    for name, key in [("climate_zone", base["in.ashrae_iecc_climate_zone_2004"].astype(str)),
                      ("state", base["in.state"].astype(str)),
                      ("county", base["in.county_and_puma"].astype(str).str.split(",").str[0]),
                      ("puma", base["in.county_and_puma"].astype(str))]:
        m = x.groupby(key).transform("mean")
        report["gtc_r2_by"][name] = float(1 - ((x - m) ** 2).mean() / x.var(ddof=0))

    gcols = [A.ACTIONS.index(k) for k in GSHP]
    full = evaluate("all22", base, val, w, role, groups, ev_mask, n_est, t0, (gcols, A.ACTIONS))
    keep_cols = [j for j in range(len(A.ACTIONS)) if j not in gcols]
    no_g = evaluate("no_gshp16", base, val[:, keep_cols], w, role, groups, ev_mask, n_est, t0,
                    ([], [A.ACTIONS[j] for j in keep_cols]))
    g_full, g_no = full["gtc_gain_over_bills"], no_g["gtc_gain_over_bills"]
    dec = ("GSHP_ONLY" if g_no["ci"][1] < 0.05 else
           "GENERAL_VALUE" if g_no["ci"][0] >= 0.05 else "MIXED")
    report.update({"all22": full, "no_gshp16": no_g, "decision": dec,
                   "gtc_gain_kg_ratio_no_gshp_over_all": g_no["kg"] / g_full["kg"]})
    if not smoke:
        report["integrity"] = {"V_FIX_SCZ": full["V"]["FIX_SCZ"], "V_ORACLE": full["V"]["ORACLE"],
                               "C": {t: full["C"][t] for t in ("BROAD", "BILLS", "S0", "RICH")},
                               "match": (abs(full["V"]["FIX_SCZ"] - 2555.6) < 0.1 and
                                         abs(full["V"]["ORACLE"] - 3002.9) < 0.1 and
                                         abs(full["C"]["S0"] - 0.788) < 0.01 and abs(full["C"]["RICH"] - 0.908) < 0.01)}
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"decision {dec}; gshp-household share of gtc gain {full['gtc_gain_share_gshp_households']:.3f}; "
        f"kg ratio no-GSHP/all {report['gtc_gain_kg_ratio_no_gshp_over_all']:.3f}; "
        f"integrity {report.get('integrity', {}).get('match')}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "run"])
    run(smoke=ap.parse_args().stage == "smoke")
