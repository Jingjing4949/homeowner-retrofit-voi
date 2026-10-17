"""H-B1 replication of the H-A5 four-tier decomposition on ResStock 2024.2 (AMY2018), see H_B1_PROTOCOL_20261016_CN.md.

Reuses the frozen H-A5 code; only the dataset, action list, emission scenarios and GSHP action ids change.
"""
import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

import h_a0 as A
import h_a1 as H
import h_a2
import h_a5

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache_b1"
S3 = ("oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
      "2024/resstock_amy2018_release_2/metadata_and_annual_results/national/parquet/")
ACTIONS = list(range(1, 17))
GSHP_IDS = (5, 10, 15)
SCEN_SRC = {"lrmer_mid_15": "out.emissions.all_fuels.lrmer_mid_case_15.co2e_kg",
            "lrmer_high_re_15": "out.emissions.all_fuels.lrmer_high_re_cost_15.co2e_kg",
            "lrmer_low_re_15": "out.emissions.all_fuels.lrmer_low_re_cost_15.co2e_kg"}
RENAME = {"in.sqft": "in.sqft..ft2", "out.bills.all_fuels.usd": A.BILL,
          "out.electricity.total.energy_consumption.kwh": A.FUELS["base_elec_kwh"],
          "out.natural_gas.total.energy_consumption.kwh": A.FUELS["base_gas_kwh"],
          "out.propane.total.energy_consumption.kwh": A.FUELS["base_propane_kwh"],
          "out.fuel_oil.total.energy_consumption.kwh": A.FUELS["base_oil_kwh"],
          **{v: h_a2.SCENARIOS[k] for k, v in SCEN_SRC.items()}}


def fetch():
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    CACHE.mkdir(exist_ok=True)
    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    manifest = {}
    inv = {v: k for k, v in RENAME.items()}
    for k in [0] + ACTIONS:
        out = CACHE / f"upgrade{k}.parquet"
        src = "baseline" if k == 0 else f"upgrade{k:02d}"
        cols = ["bldg_id", "applicability", A.BILL] + list(h_a2.SCENARIOS[s] for s in SCEN_SRC)
        if k == 0:
            cols += ["weight"] + A.IN_COLS + list(A.FUELS.values())
        if not out.exists():
            t = time.time()
            tb = pq.read_table(S3 + f"{src}_metadata_and_annual_results.parquet",
                               columns=[inv.get(c, c) for c in cols], filesystem=s3)
            tb = tb.rename_columns([RENAME.get(c, c) for c in tb.column_names])
            pq.write_table(tb, out)
            print(f"{src}: {tb.num_rows} rows, {time.time() - t:.1f}s", flush=True)
        manifest[out.name] = {"sha256": A.sha256(out), "bytes": out.stat().st_size}
    (CACHE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


def configure():
    A.CACHE = CACHE
    A.ACTIONS = ACTIONS
    h_a2.ALT_DIR = CACHE
    h_a2.SCENARIOS = {k: h_a2.SCENARIOS[k] for k in SCEN_SRC}
    src = inspect.getsource(h_a5.run_scenario)
    assert src.count("(6, 7, 8, 18, 31, 32)") == 1
    h_a5.GSHP_IDS = GSHP_IDS
    exec(src.replace("(6, 7, 8, 18, 31, 32)", "GSHP_IDS"), h_a5.__dict__)


def run(smoke):
    configure()
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
        out_dir, eval_mask = ROOT / "smoke_output_b1", sel
    else:
        out_dir, eval_mask = ROOT / "output_b1", role == "test"
    out_dir.mkdir(exist_ok=True)
    H.log(f"device {dev}; n {len(base)}; eval {eval_mask.sum()}; actions {A.ACTIONS}; GSHP {GSHP_IDS}", t0)
    ids = base["bldg_id"].to_numpy()
    report = {"smoke": smoke, "dataset": S3, "checks": checks, "cfg": dict(H.CFG), "tiers": h_a5.TIERS,
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
    report["verdict"] = {k: all(res[s]["decomposition"][k] for s in h_a2.SCENARIOS)
                         for k in ["R1_home", "R2_short_questionnaire", "R3_static_sufficient", "R4_targeted_test"]}
    report["elapsed_s"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    H.log(f"verdict {report['verdict']}", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "smoke", "run"])
    st = ap.parse_args().stage
    fetch() if st == "fetch" else run(smoke=st == "smoke")
