"""Figures for the manuscript, drawn from the registered held-out results (runs/remote_a6, runs/remote_b2)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = {"A": ROOT / "runs/remote_a6/output_a6/summary.json", "B": ROOT / "runs/remote_b2/output_b2/summary.json"}
SC = "lrmer_mid_15"
MM = 1 / 25.4
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "grey": "#7F7F7F", "sky": "#56B4E9"}
NAMES = {"A": "Dataset A (ResStock 2025 R1)", "B": "Dataset B (ResStock 2024.2)"}

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

data = {k: json.load(open(p))["results"][SC] for k, p in RES.items()}


def pt(d, key):
    c, (lo, hi) = d["C"][key], d["C_ci"][key]
    return c, c - lo, hi - c


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_tiers():
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM), sharey=True)
    xl = ["Records\n(T1)", "+ Homeowner\n(T2)", "+ Audit\n(T3)", "+ Ground test\n(T4)"]
    gbm = ["GBM_T1_RECORDS", "GBM_T2_HOME", "GBM_T3_AUDIT", "GBM_T4_TEST"]
    nn = ["HOME_STATIC_CZ@0", "NN_HOME", "NN_FULL"]
    for ax, (k, d) in zip(axes, data.items()):
        for keys, xs, col, mk, lab, dx in [(nn, [0, 1, 2], OI["blue"], "o", "Value network", -0.06),
                                           (gbm, [0, 1, 2, 3], OI["orange"], "s", "Gradient boosting", 0.06)]:
            ps = [pt(d, q) for q in keys]
            x = [i + dx for i in xs]
            ax.errorbar(x, [p[0] for p in ps], yerr=[[p[1] for p in ps], [p[2] for p in ps]], color=col, marker=mk,
                        ms=4.5, lw=1.2, capsize=2.5, elinewidth=0.8, label=lab)
            for xi, p in zip(x, ps):
                ax.annotate(f"{p[0]:.2f}", (xi, p[0]), xytext=(0, 9) if dx < 0 else (7, -7),
                            textcoords="offset points", fontsize=6.5, color=col,
                            ha="center" if dx < 0 else "left", va="center")
        ax.set_xticks(range(4), xl)
        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(0, 1)
        ax.set_title(NAMES[k], loc="left", fontweight="bold")
        ax.grid(axis="y", lw=0.4, color="#dddddd")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recovery rate C (0 = climate-zone default, 1 = oracle)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=2)
    save(fig, "fig1_tiers")


def fig_questions():
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM))
    ks = range(5)
    series = [("HOME_STATIC_CZ", "Fixed order per climate zone, homeowner items", OI["blue"], "o", "-"),
              ("HOME_EVOI", "Adaptive EVOI, homeowner items", OI["green"], "^", "-"),
              ("STATIC_GREEDY_CZ", "Fixed order per climate zone, any item", OI["purple"], "D", "--"),
              ("ENTROPY", "Most uncertain answer first, any item", OI["grey"], "v", ":")]
    for ax, (k, d) in zip(axes, data.items()):
        lo, hi = zip(*[d["C_ci"][f"HOME_STATIC_CZ@{i}"] for i in ks])
        ax.fill_between(ks, lo, hi, color=OI["blue"], alpha=0.15, lw=0)
        for key, lab, col, mk, ls in series:
            ax.plot(ks, [d["C"][f"{key}@{i}"] for i in ks], color=col, marker=mk, ms=3.8, lw=1.2, ls=ls, label=lab)
        for key, lab, ls in [("NN_HOME", "All 15 homeowner items", (0, (4, 2))),
                             ("NN_FULL", "All homeowner + audit items", (0, (1, 1.5)))]:
            ax.axhline(d["C"][key], color="k", lw=0.8, ls=ls, label=lab)
        r3 = d["decomposition"]["rho3_home_static"]
        c0 = d["C"]["HOME_STATIC_CZ@0"]
        ax.set_ylim(c0 - 0.04, d["C"]["NN_FULL"] + 0.01)
        ax.annotate(f"3 questions: ρ3 = {r3['est']:.2f} [{r3['ci'][0]:.2f}, {r3['ci'][1]:.2f}]\nof the all-homeowner-item value",
                    (3, d["C"]["HOME_STATIC_CZ@3"]), xytext=(1.6, c0 - 0.006), va="top",
                    fontsize=6.5, color=OI["blue"], arrowprops=dict(arrowstyle="-", color=OI["blue"], lw=0.6))
        ax.set_xticks(list(ks))
        ax.set_xlabel("Number of questions asked")
        ax.set_title(NAMES[k], loc="left", fontweight="bold")
        ax.grid(axis="y", lw=0.4, color="#dddddd")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Recovery rate C")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2)
    save(fig, "fig2_questions")


# GSHP package applicability by ground thermal conductivity level, weighted, all occupied dwellings:
# paper/_verification/gshp_applicability_by_conductivity_CN.md (ranges over the GSHP packages of each release).
LEVELS = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]
APP = {"A": [(0, 0), (0, 0), (78.2, 78.6), (75.0, 75.5), (73.1, 73.6), (71.4, 71.7), (72.2, 72.9), (71.7, 72.0)],
       "B": [(6.5, 7.3), (69.1, 69.1), (77.1, 77.1), (73.7, 73.7), (71.2, 71.2), (69.2, 69.2), (69.8, 69.8),
             (71.7, 71.7)]}
SHARE_A = [0.6, 7.1, 45.7, 36.0, 7.9, 2.4, 0.2, 0.2]


def fig_ground():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(190 * MM, 68 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
    x = range(len(LEVELS))
    ax.bar(x, SHARE_A, color="#e6e6e6", width=0.7, label="Share of dwellings (%)")
    for k, col, mk, dx in [("A", OI["red"], "o", -0.08), ("B", OI["blue"], "s", 0.08)]:
        mid = [(a + b) / 2 for a, b in APP[k]]
        ax.plot([i + dx for i in x], mid, color=col, marker=mk, ms=4, lw=1.2,
                label=f"GSHP packages applicable (%), dataset {k}")
    ax.set_xticks(list(x), [f"{v:.1f}" for v in LEVELS])
    ax.set_xlabel("Ground thermal conductivity, Btu/(h·ft·°F)")
    ax.set_ylabel("Percent of dwellings")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("(a) GSHP applicability by conductivity level", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    d = data["A"]["decomposition"]
    qs = [0.1, 0.2, 0.3, 0.5]
    ev = [d[f"phi_evoi@{q}"] for q in qs]
    diff = [d[f"phi_evoi_minus_random@{q}"]["est"] for q in qs]
    bx.errorbar([q * 100 for q in qs], [e["est"] for e in ev],
                yerr=[[e["est"] - e["ci"][0] for e in ev], [e["ci"][1] - e["est"] for e in ev]], color=OI["red"],
                marker="o", ms=4, lw=1.2, capsize=2.5, elinewidth=0.8, label="Targeted by expected value of testing")
    bx.plot([q * 100 for q in qs], [e["est"] - df for e, df in zip(ev, diff)], color=OI["grey"], marker="s", ms=3.5,
            lw=1.0, ls="--", label="Random dwellings")
    bx.set_xlabel("Dwellings tested (%)")
    bx.set_ylabel("Share of full-testing value recovered")
    bx.set_xlim(0, 55)
    bx.set_ylim(0, 1)
    bx.set_title("(b) Targeted ground testing, dataset A", loc="left", fontweight="bold")
    bx.legend(loc="lower right", frameon=False)
    for a in (ax, bx):
        a.grid(axis="y", lw=0.4, color="#dddddd")
        a.set_axisbelow(True)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_ground")


def fig_graphical_abstract():
    fig = plt.figure(figsize=(13.28 / 2.54 * 1.6, 5.31 / 2.54 * 1.6))
    fig.text(0.015, 0.94, "What should a homeowner be asked before a retrofit is recommended?", fontsize=10,
             fontweight="bold", va="top")
    fig.text(0.015, 0.84, "Counterfactual simulations of every retrofit package for ~483,000 dwellings in each of "
             "two independent ResStock releases", fontsize=7.5, va="top", color="#444444")
    ax = fig.add_axes([0.07, 0.15, 0.36, 0.55])
    bx = fig.add_axes([0.58, 0.15, 0.38, 0.55])
    w = 0.36
    for i, (k, d) in enumerate(data.items()):
        x = d["decomposition"]
        for j, (key, col, lab) in enumerate([("nn_home_gain", OI["blue"], "15 homeowner questions"),
                                             ("nn_audit_gain", OI["orange"], "10 audit measurements added afterwards")]):
            v = x[key]["est"] * 100
            ax.bar(i + (j - 0.5) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (j - 0.5) * w, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], ["Dataset A", "Dataset B"])
    ax.set_ylabel("Percentage points of value gap\nrecovered", fontsize=7.5)
    ax.set_ylim(0, 21)
    ax.legend(loc="upper left", frameon=False, fontsize=6.8)
    ax.set_title("Homeowner answers beat the audit", loc="left", fontsize=8.5, fontweight="bold")
    for k, col, mk in [("A", OI["red"], "o"), ("B", OI["blue"], "s")]:
        d = data[k]["C"]
        c0, ch = d["HOME_STATIC_CZ@0"], d["NN_HOME"]
        bx.plot(range(5), [(d[f"HOME_STATIC_CZ@{i}"] - c0) / (ch - c0) * 100 for i in range(5)], color=col,
                marker=mk, ms=4, lw=1.3, label=f"Dataset {k}")
    bx.axvline(3, color="#999999", lw=0.6, ls=":")
    bx.set_xticks(range(5))
    bx.set_xlabel("Homeowner questions asked (fixed order)", fontsize=7.5)
    bx.set_ylabel("% of all-15-question value", fontsize=7.5)
    bx.set_ylim(0, 100)
    bx.legend(loc="lower right", frameon=False, fontsize=7)
    bx.set_title("Three fixed questions give ~80%", loc="left", fontsize=8.5, fontweight="bold")
    for a in (ax, bx):
        a.grid(axis="y", lw=0.4, color="#dddddd")
        a.set_axisbelow(True)
    fig.savefig(OUT / "graphical_abstract.pdf")
    fig.savefig(OUT / "graphical_abstract.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_tiers()
    fig_questions()
    fig_ground()
    fig_graphical_abstract()
