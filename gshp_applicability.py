"""GSHP package applicability and mean emission reduction by ground thermal conductivity level (Fig. 3a)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EMIS = "out.emissions.total.lrmer_mid_case_15..co2e_kg"
GTC = "in.ground_thermal_conductivity"
DATA = {"A": (ROOT / "cache", [6, 7, 8, 18, 31, 32], 6), "B": (ROOT / "cache_b1", [5, 10, 15], 5)}


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def release(cache, gshp, single):
    base = pd.read_parquet(cache / "upgrade0.parquet", columns=["bldg_id", "weight", GTC, EMIS, "in.vacancy_status"])
    base = base[base["in.vacancy_status"] == "Occupied"].set_index("bldg_id")
    base["level"] = base[GTC].str.extract(r"([\d.]+)")[0].astype(float)
    ups = {k: pd.read_parquet(cache / f"upgrade{k}.parquet", columns=["bldg_id", "applicability", EMIS])
           .set_index("bldg_id").reindex(base.index) for k in gshp}
    w = base["weight"].to_numpy()
    out = {}
    for lv, idx in base.groupby("level").indices.items():
        ap = [wmean(ups[k]["applicability"].to_numpy()[idx].astype(float), w[idx]) for k in gshp]
        u = ups[single].iloc[idx]
        ok = u["applicability"].to_numpy().astype(bool)
        red = base[EMIS].to_numpy()[idx][ok] - u[EMIS].to_numpy()[ok]
        out[f"{lv:.1f}"] = {"share": float(w[idx].sum() / w.sum()), "applicable_min": min(ap),
                            "applicable_max": max(ap),
                            "mean_reduction_single_gshp_kg": wmean(red, w[idx][ok]) if ok.any() else None}
    return out


if __name__ == "__main__":
    res = {k: release(*v) for k, v in DATA.items() if (v[0] / "upgrade0.parquet").exists()}
    json.dump(res, sys.stdout, indent=1)
