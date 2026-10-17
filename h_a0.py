"""H-A0 headroom check on ResStock 2025 R1 (see H_A0_PROTOCOL_20261014_CN.md)."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=lgb.basic.LGBMDeprecationWarning)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
S3_PREFIX = ("oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
             "2025/resstock_amy2018_release_1/metadata_and_annual_results/national/full/parquet/")
SALT = "HA0-20261014"

ACTIONS = list(range(1, 19)) + [29, 30, 31, 32]
SINGLE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 29]

BILL = "out.utility_bills.total_bill..usd"
EMIS = "out.emissions.total.lrmer_mid_case_15..co2e_kg"
FUELS = {
    "base_elec_kwh": "out.electricity.total.energy_consumption..kwh",
    "base_gas_kwh": "out.natural_gas.total.energy_consumption..kwh",
    "base_propane_kwh": "out.propane.total.energy_consumption..kwh",
    "base_oil_kwh": "out.fuel_oil.total.energy_consumption..kwh",
}

CONTEXT = ["in.state", "in.ashrae_iecc_climate_zone_2004"]
BROAD = CONTEXT + [
    "in.geometry_building_type_recs", "in.vintage", "in.sqft..ft2", "in.geometry_stories_low_rise",
    "in.heating_fuel", "in.hvac_heating_type", "in.hvac_cooling_type", "in.water_heater_fuel",
]
BILLS = BROAD + ["base_bill"] + list(FUELS)
RICH = BILLS + [
    "in.geometry_building_type_height", "in.hvac_heating_efficiency", "in.hvac_cooling_efficiency",
    "in.hvac_has_ducts", "in.duct_leakage_and_insulation", "in.duct_location", "in.insulation_wall",
    "in.insulation_ceiling", "in.insulation_floor", "in.insulation_foundation_wall", "in.insulation_roof",
    "in.insulation_slab", "in.infiltration", "in.windows", "in.window_areas", "in.water_heater_efficiency",
    "in.water_heater_location", "in.geometry_attic_type", "in.geometry_foundation_type",
    "in.geometry_wall_type", "in.heating_setpoint", "in.cooling_setpoint", "in.hvac_has_shared_system",
    "in.ground_thermal_conductivity", "in.occupants", "in.usage_level",
]
TIERS = {"BROAD": BROAD, "BILLS": BILLS, "RICH": RICH}
NUMERIC = {"in.sqft..ft2", "base_bill", *FUELS}
IN_COLS = sorted({c for c in RICH if c.startswith("in.")} | {"in.county_and_puma", "in.vacancy_status"})

LGB_PARAMS = dict(objective="regression", num_leaves=63, learning_rate=0.05, min_data_in_leaf=100,
                  feature_fraction=0.9, n_estimators=2000, verbose=-1, n_jobs=8)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch():
    CACHE.mkdir(exist_ok=True)
    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    manifest = {}
    for k in [0] + ACTIONS:
        out = CACHE / f"upgrade{k}.parquet"
        cols = ["bldg_id", "applicability", BILL, EMIS]
        if k == 0:
            cols += ["weight"] + IN_COLS + list(FUELS.values())
        if not out.exists():
            t = time.time()
            tb = pq.read_table(S3_PREFIX + f"upgrade{k}.parquet", columns=cols, filesystem=s3)
            pq.write_table(tb, out)
            print(f"upgrade{k}: {tb.num_rows} rows, {time.time() - t:.1f}s", flush=True)
        manifest[out.name] = {"sha256": sha256(out), "bytes": out.stat().st_size}
    (CACHE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


def bucket(groups, salt, mod):
    return np.array([int(hashlib.sha256((salt + g).encode()).hexdigest(), 16) % mod for g in groups])


def load():
    base = pd.read_parquet(CACHE / "upgrade0.parquet")
    base = base.rename(columns={BILL: "base_bill", EMIS: "base_emis", **{v: k for k, v in FUELS.items()}})
    ups = {k: pd.read_parquet(CACHE / f"upgrade{k}.parquet") for k in ACTIONS}
    common = set(base["bldg_id"])
    for u in ups.values():
        common &= set(u["bldg_id"])
    n_before = len(base)
    base = base[base["bldg_id"].isin(common)].sort_values("bldg_id").reset_index(drop=True)
    ids = base["bldg_id"].to_numpy()
    n, a = len(base), len(ACTIONS)
    val, db, de, app = (np.zeros((n, a)) for _ in range(4))
    checks = {"dropped_not_in_all_files": int(n_before - n)}
    for j, k in enumerate(ACTIONS):
        u = ups[k][ups[k]["bldg_id"].isin(common)].sort_values("bldg_id").reset_index(drop=True)
        assert np.array_equal(u["bldg_id"].to_numpy(), ids), f"bldg_id mismatch upgrade{k}"
        ap = u["applicability"].to_numpy().astype(bool)
        dB = base["base_bill"].to_numpy() - u[BILL].to_numpy()
        dE = base["base_emis"].to_numpy() - u[EMIS].to_numpy()
        na = ~ap
        rel = np.abs(dB[na]) / np.maximum(np.abs(base["base_bill"].to_numpy()[na]), 1e-9)
        rel_e = np.abs(dE[na]) / np.maximum(np.abs(base["base_emis"].to_numpy()[na]), 1e-9)
        checks[str(k)] = {"applicable_frac": float(ap.mean()),
                          "nonapp_deviating_frac": float(((rel > 1e-6) | (rel_e > 1e-6)).mean()) if na.any() else 0.0}
        db[:, j], de[:, j], app[:, j] = dB, dE, ap
        val[:, j] = np.where(ap & (dB >= 0), dE, 0.0)
    keep = (base["in.vacancy_status"] == "Occupied").to_numpy()
    for c in RICH:
        if c not in NUMERIC:
            base[c] = base[c].astype("string").fillna("__MISSING__").astype("category")
    checks["missing_rate"] = {c: float(base[c].isna().mean()) for c in RICH if c in NUMERIC}
    checks["n_total"], checks["n_occupied"] = int(n), int(keep.sum())
    return base[keep].reset_index(drop=True), val[keep], db[keep], de[keep], app[keep], checks


def split(base):
    g = base["in.county_and_puma"].astype(str).to_numpy()
    b = bucket(g, SALT, 100)
    role = np.where(b < 60, "train", np.where(b < 80, "tune", "test"))
    es = (role == "train") & (bucket(g, SALT + "-ES", 10) == 0)
    role = np.where(es, "es", role)
    return role, g


def fixed_policy(base, val, w, fit_mask, keys, min_n=100):
    """Per-cell argmax of weighted mean value, fallback through `keys` (most to least specific)."""
    vals = np.concatenate([np.zeros((len(val), 1)), val], axis=1)
    glob = int(np.argmax((vals[fit_mask] * w[fit_mask, None]).sum(0)))
    choice = np.full(len(base), glob)
    assigned = np.zeros(len(base), bool)
    for key in keys:
        cell = base[key].astype(str).agg("|".join, axis=1).to_numpy()
        df = pd.DataFrame(vals[fit_mask] * w[fit_mask, None]).assign(c=cell[fit_mask], w=w[fit_mask], n=1)
        agg = df.groupby("c").sum()
        ok = agg[agg["n"] >= min_n]
        best = dict(zip(ok.index, np.argmax(ok.drop(columns=["w", "n"]).to_numpy(), axis=1)))
        hit = np.array([c in best for c in cell]) & ~assigned
        choice[hit] = [best[c] for c in cell[hit]]
        assigned |= hit
    return choice  # 0 = no retrofit, j+1 = action column j


def realized(val, choice):
    vals = np.concatenate([np.zeros((len(val), 1)), val], axis=1)
    return vals[np.arange(len(val)), choice]


def fit_predict(base, val, w, role, feats, eval_mask, n_estimators):
    X = base[feats]
    tr, es = role == "train", role == "es"
    pred = np.zeros((int(eval_mask.sum()), val.shape[1]))
    iters = []
    for j in range(val.shape[1]):
        m = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_estimators})
        m.fit(X[tr], val[tr, j], sample_weight=w[tr], eval_set=[(X[es], val[es, j])],
              eval_sample_weight=[w[es]], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred[:, j] = m.predict(X[eval_mask])
        iters.append(int(m.best_iteration_ or n_estimators))
    return pred, iters


def choose(pred):
    best = np.argmax(pred, axis=1)
    return np.where(pred[np.arange(len(pred)), best] > 0, best + 1, 0)


def cluster_boot(groups, w, values, n_boot=1000, seed=20261014):
    ug, gi = np.unique(groups, return_inverse=True)
    W = np.bincount(gi, weights=w)
    S = {k: np.bincount(gi, weights=w * v) for k, v in values.items()}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in values}
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        tw = (cnt * W).sum()
        for k in values:
            out[k].append((cnt * S[k]).sum() / tw)
    return {k: np.array(v) for k, v in out.items()}


def summarize(V, boot, fix="FIX_SCZ"):
    G = V["ORACLE"] - V[fix]
    res = {"V": V, "G": G, "r": G / V["ORACLE"], "C": {}}
    bG = boot["ORACLE"] - boot[fix]
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res["G_ci"], res["r_ci"] = ci(bG), ci(bG / boot["ORACLE"])
    bC = {}
    for t in TIERS:
        k = f"GBM_{t}"
        if k in V:
            res["C"][t] = (V[k] - V[fix]) / G
            bC[t] = (boot[k] - boot[fix]) / bG
            res[f"C_{t}_ci"] = ci(bC[t])
    if "RICH" in bC and "BROAD" in bC:
        res["C_RICH_minus_BROAD"] = res["C"]["RICH"] - res["C"]["BROAD"]
        res["C_RICH_minus_BROAD_ci"] = ci(bC["RICH"] - bC["BROAD"])
    return res


def decide(s):
    if s["r"] < 0.10:
        return ["NO_HEADROOM"]
    flags = []
    if s["C_RICH_minus_BROAD_ci"][0] >= 0.10:
        flags.append("QUESTION_VALUE")
    if 1 - s["C"]["RICH"] >= 0.20:
        flags.append("DECISION_GAP")
    if not flags and s["C"]["BROAD"] >= 0.90:
        flags.append("BROAD_SUFFICIENT")
    return flags or ["WEAK_HEADROOM"]


def run(smoke):
    t0 = time.time()
    base, val, db, de, app, checks = load()
    role, groups = split(base)
    if smoke:
        # smoke test: CO only, evaluate on the early-stopping holdout; never touches tune/test rows
        co = (base["in.state"] == "CO").to_numpy()
        base, val, db, role, groups = base[co].reset_index(drop=True), val[co], db[co], role[co], groups[co]
        g = groups
        sub = bucket(g, SALT + "-SMOKE", 2)
        role = np.where(role == "train", np.where(sub == 0, "train", "es"), "excluded")
        eval_mask, n_est, out_dir = role == "es", 50, ROOT / "smoke_output"
    else:
        eval_mask, n_est, out_dir = role == "tune", LGB_PARAMS["n_estimators"], ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    w = base["weight"].to_numpy()
    fit_mask = role == "train"
    ev = eval_mask
    report = {"checks": checks, "roles": {r: int((role == r).sum()) for r in np.unique(role)},
              "n_eval": int(ev.sum()), "smoke": smoke}

    for name, cols in [("primary", list(range(len(ACTIONS)))),
                       ("single", [ACTIONS.index(k) for k in SINGLE])]:
        v, dbs = val[:, cols], db[:, cols]
        acts = [ACTIONS[c] for c in cols]
        choices = {
            "FIX_GLOBAL": fixed_policy(base, v, w, fit_mask, []),
            "FIX_SCZ": fixed_policy(base, v, w, fit_mask, [["in.state", "in.ashrae_iecc_climate_zone_2004"],
                                                           ["in.ashrae_iecc_climate_zone_2004"]]),
        }
        choices = {k: c[ev] for k, c in choices.items()}
        iters = {}
        for t, feats in TIERS.items():
            pred, iters[t] = fit_predict(base, v, w, role, feats, ev, n_est)
            choices[f"GBM_{t}"] = choose(pred)
            print(f"[{name}] {t} done at {time.time() - t0:.0f}s", flush=True)
        vev = v[ev]
        oracle_val = np.maximum(vev.max(1), 0)
        values = {k: realized(vev, c) for k, c in choices.items()}
        values["ORACLE"] = oracle_val
        V = {k: float(np.average(x, weights=w[ev])) for k, x in values.items()}
        boot = cluster_boot(groups[ev], w[ev], values)
        s = summarize(V, boot)
        s["decision"] = decide(s) if name == "primary" else "descriptive_only"
        oracle_choice = np.where(vev.max(1) > 0, np.argmax(vev, 1) + 1, 0)
        labels = ["none"] + [str(a) for a in acts]
        s["oracle_action_share"] = {labels[i]: float(np.average(oracle_choice == i, weights=w[ev]))
                                    for i in range(len(labels))}
        dbe = dbs[ev]
        s["bill_increase_rate"] = {}
        for k, c in choices.items():
            rows = c > 0
            inc = np.zeros(len(c), bool)
            inc[rows] = dbe[np.where(rows)[0], c[rows] - 1] < 0
            s["bill_increase_rate"][k] = float(np.average(inc, weights=w[ev]))
        s["recommendation_share"] = {k: {labels[i]: float(np.average(c == i, weights=w[ev]))
                                         for i in np.unique(c)} for k, c in choices.items()}
        s["best_iteration"] = iters
        report[name] = s
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("roles", "n_eval")}, indent=1))
    p = report["primary"]
    print("V:", {k: round(x, 1) for k, x in p["V"].items()})
    print("r:", round(p["r"], 4), p["r_ci"], "C:", {k: round(x, 3) for k, x in p["C"].items()},
          "decision:", p["decision"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "smoke", "run"])
    a = ap.parse_args()
    if a.stage == "fetch":
        fetch()
    else:
        run(smoke=a.stage == "smoke")
