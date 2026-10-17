"""H-A6 homeowner-questionnaire evaluation on the held-out parts of both datasets (see H_A6_PROTOCOL_20261016_CN.md).

Runs the frozen H-A5 scenario code unchanged and adds network evaluations restricted to the 15 homeowner items.
Dataset A: ResStock 2025 R1, sealed test split (used once before in H-A3). Dataset B: ResStock 2024.2, test split
(used once before in H-B1).
"""
import inspect
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

ROOT = Path(__file__).resolve().parent
NH = len(h_a5.HOME)
assert H.Q[:NH] == h_a5.HOME
CLAIMS = ["Q1_home_value", "Q2_home_over_audit", "Q3_short_home_questionnaire"]
REF = {"A": {"lrmer_mid_15": (2587.97, 3047.21), "lrmer_high_re_15": (2689.39, 3172.35),
             "lrmer_low_re_15": (2568.58, 3028.18), "aer_mid_avg": (2493.60, 2925.25)},
       "B": {"lrmer_mid_15": (2759.38, 2948.17), "lrmer_high_re_15": (2892.99, 3080.32),
             "lrmer_low_re_15": (2657.80, 2842.85)}}


def static_greedy_home(eng, rows, val, w, kmax):
    order = []
    for _ in range(kmax):
        best_q, best_v = None, -np.inf
        for q in range(NH):
            if q in order:
                continue
            obs = torch.zeros(len(rows), H.NQ, dtype=torch.bool, device=eng.dev)
            obs[:, order + [q]] = True
            v = H.wavg(A.realized(val, eng.decide(rows, obs).cpu().numpy()), w)
            if v > best_v:
                best_q, best_v = q, v
        order.append(best_q)
    return order


def home_block(eng, rows, ev, vev, wev, val, w, sel, base, dev, values, out, name, t0, kmax):
    n = len(ev)
    obs = torch.zeros(n, H.NQ, dtype=torch.bool, device=dev)
    obs[:, :NH] = True
    values["NN_HOME"] = A.realized(vev, eng.decide(rows, obs).cpu().numpy())
    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()
    order_g = static_greedy_home(eng, torch.tensor(np.where(sel)[0], device=dev), val[sel], w[sel], kmax)
    zone_orders = {z: static_greedy_home(eng, torch.tensor(np.where(sel & (zone == z))[0], device=dev),
                                         val[sel & (zone == z)], w[sel & (zone == z)], kmax)
                   for z in np.unique(zone[sel]) if (sel & (zone == z)).sum() >= H.MIN_ZONE_ROWS}
    uz, zi = np.unique(zone[ev], return_inverse=True)
    orders = torch.tensor(np.array([zone_orders.get(z, order_g) for z in uz])[zi], device=dev)
    out["home_static_order"] = [H.Q[q] for q in order_g]
    out["home_zone_orders"] = {z: [H.Q[q] for q in o] for z, o in zone_orders.items()}
    ar = torch.arange(n, device=dev)
    obs_s = torch.zeros(n, H.NQ, dtype=torch.bool, device=dev)
    obs_e = torch.zeros(n, H.NQ, dtype=torch.bool, device=dev)
    forbid = torch.zeros(n, H.NQ, dtype=torch.bool, device=dev)
    forbid[:, NH:] = True
    values["HOME_STATIC_CZ@0"] = values["HOME_EVOI@0"] = A.realized(vev, eng.decide(rows, obs_s).cpu().numpy())
    asked = []
    for k in range(kmax):
        obs_s[ar, orders[:, k]] = True
        values[f"HOME_STATIC_CZ@{k + 1}"] = A.realized(vev, eng.decide(rows, obs_s).cpu().numpy())
        s, _ = eng.adaptive_scores(rows, obs_e, "EVOI")
        q = s.masked_fill(obs_e | forbid, float("-inf")).argmax(1)
        obs_e[ar, q] = True
        asked.append(q)
        values[f"HOME_EVOI@{k + 1}"] = A.realized(vev, eng.decide(rows, obs_e).cpu().numpy())
    asked = torch.stack(asked, 1).cpu().numpy()
    out["home_question_share_evoi"] = {H.Q[q]: float(np.average((asked == q).any(1), weights=wev)) for q in range(NH)}
    H.log(f"[{name}] NN_HOME V {H.wavg(values['NN_HOME'], wev):.1f}; HOME_STATIC_CZ "
          f"{[round(H.wavg(values[f'HOME_STATIC_CZ@{k}'], wev), 1) for k in range(kmax + 1)]}; HOME_EVOI "
          f"{[round(H.wavg(values[f'HOME_EVOI@{k}'], wev), 1) for k in range(kmax + 1)]}", t0)


def home_stats(stat):
    ks = h_a5.K_LEVELS
    d = {
        "nn_home_gain": stat(lambda c: c["NN_HOME"] - c["EVOI@0"]),
        "nn_audit_gain": stat(lambda c: c["NN_FULL"] - c["NN_HOME"]),
        "home_minus_audit": stat(lambda c: 2 * c["NN_HOME"] - c["EVOI@0"] - c["NN_FULL"]),
        "audit_share": stat(lambda c: (c["NN_FULL"] - c["NN_HOME"]) / (c["NN_FULL"] - c["EVOI@0"])),
        "rho3_home_static": stat(lambda c: (c["HOME_STATIC_CZ@3"] - c["EVOI@0"]) / (c["NN_HOME"] - c["EVOI@0"])),
        "rho3_home_evoi": stat(lambda c: (c["HOME_EVOI@3"] - c["EVOI@0"]) / (c["NN_HOME"] - c["EVOI@0"])),
        "D_home": stat(lambda c: np.mean([c[f"HOME_EVOI@{k}"] - c[f"HOME_STATIC_CZ@{k}"] for k in ks], 0)),
        "home3_minus_any3_static": stat(lambda c: c["HOME_STATIC_CZ@3"] - c["STATIC_GREEDY_CZ@3"]),
    }
    d["Q1_home_value"] = d["nn_home_gain"]["ci"][0] > 0
    d["Q2_home_over_audit"] = d["home_minus_audit"]["ci"][0] > 0
    d["Q3_short_home_questionnaire"] = d["rho3_home_static"]["ci"][0] >= 0.5
    return d


def patch(gshp_ids):
    src = inspect.getsource(h_a5.run_scenario)
    anchors = {
        '    zone = base["in.ashrae_iecc_climate_zone_2004"].astype(str).to_numpy()\n    kmax = max(K_LEVELS)\n':
            '    home_block(eng, rows, ev, vev, wev, val, w, sel, base, dev, values, out, name, t0, max(K_LEVELS))\n',
        '    lo_s, hi_s, lo_i = d["D_static"]["ci"][0]': '    d.update(home_stats(stat))\n',
        "(6, 7, 8, 18, 31, 32)": None,
    }
    for a, ins in anchors.items():
        assert src.count(a) == 1, a
        src = src.replace(a, str(gshp_ids)) if ins is None else src.replace(a, ins + a)
    h_a5.home_block, h_a5.home_stats = home_block, home_stats
    exec(src, h_a5.__dict__)


def run(dataset, smoke):
    if dataset == "B":
        import h_b1
        A.CACHE, A.ACTIONS, h_a2.ALT_DIR = h_b1.CACHE, h_b1.ACTIONS, h_b1.CACHE
        h_a2.SCENARIOS = {k: h_a2.SCENARIOS[k] for k in h_b1.SCEN_SRC}
        patch(h_b1.GSHP_IDS)
    else:
        patch((6, 7, 8, 18, 31, 32))
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not smoke and dev.type != "cuda":
        raise RuntimeError("Full run requires CUDA")
    A.LGB_PARAMS["n_jobs"] = h_a2.cpu_count()
    base, val0, db, de, app, checks = A.load()
    role, groups = A.split(base)
    sel = (role == "train") & (A.bucket(groups, H.SALT + "-SEL", 10) == 0)
    tag = "a6" if dataset == "A" else "b2"
    if smoke:
        keep = np.where((base["in.state"] == "CO").to_numpy())[0]
        base, val0, db, app, role, groups, sel = (base.iloc[keep].reset_index(drop=True), val0[keep], db[keep],
                                                  app[keep], role[keep], groups[keep], sel[keep])
        H.CFG.update(epochs=2, seeds=(1,), n_boot=50)
        out_dir, eval_mask = ROOT / f"smoke_output_{tag}", sel
    else:
        out_dir, eval_mask = ROOT / f"output_{tag}", role == "test"
    out_dir.mkdir(exist_ok=True)
    H.log(f"dataset {dataset}; device {dev}; n {len(base)}; eval {eval_mask.sum()}; actions {len(A.ACTIONS)}", t0)
    ids = base["bldg_id"].to_numpy()
    report = {"smoke": smoke, "dataset": dataset, "cfg": dict(H.CFG), "home": h_a5.HOME, "audit": h_a5.AUDIT,
              "scenarios": h_a2.SCENARIOS, "results": {}}
    for name, col in h_a2.SCENARIOS.items():
        val = val0 if col == A.EMIS else h_a2.scenario_values(ids, db, app, col)
        try:
            report["results"][name] = h_a5.run_scenario(name, base, val, role, groups, sel, eval_mask, smoke, dev,
                                                        t0)
        except ZeroDivisionError:
            if not smoke:
                raise
            H.log(f"[{name}] smoke: 2-epoch network is constant, ratio undefined", t0)
            return
        (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    res = report["results"]
    keys = CLAIMS + ["R1_home", "R2_short_questionnaire", "R3_static_sufficient", "R4_targeted_test"]
    report["verdict"] = {k: all(res[s]["decomposition"][k] for s in res) for k in keys}
    if not smoke:
        report["integrity"] = {s: [res[s]["V"]["FIX_SCZ"], res[s]["V"]["ORACLE"]] for s in res}
        report["integrity"]["match"] = all(abs(res[s]["V"]["FIX_SCZ"] - REF[dataset][s][0]) < 0.1 and
                                           abs(res[s]["V"]["ORACLE"] - REF[dataset][s][1]) < 0.1 for s in res)
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"verdict {report['verdict']}; integrity {report.get('integrity', {}).get('match')}", t0)


if __name__ == "__main__":
    stage = sys.argv[1]
    assert stage in ("smoke", "run")
    run("A", stage == "smoke")
