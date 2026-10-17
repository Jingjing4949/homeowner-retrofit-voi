"""Supplementary Material: the complete record of registered analyses, read from the stored run summaries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = "lrmer_mid_15"


def load(rel):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def iv(v, nd=3):
    return f"{v['est']:.{nd}f} [{v['ci'][0]:.{nd}f}, {v['ci'][1]:.{nd}f}]".replace("-", "$-$")


def ivp(est, ci, nd=3):
    return iv({"est": est, "ci": ci}, nd)


def count(res, flag):
    vals = [r["decomposition"][flag] for r in res.values()]
    return f"{sum(vals)}/{len(vals)}"


def held(res, flag):
    return ", ".join(sc for sc, r in res.items() if r["decomposition"][flag]) or "none"


SCN = {"lrmer_mid_15": "mid", "lrmer_high_re_15": "high RE cost", "lrmer_low_re_15": "low RE cost",
       "aer_mid_avg": "AER"}


def rows():
    a1 = load("runs/remote/output/summary.json")["result"]
    a1s = load("runs/remote_a1s/output_s/summary.json")["result"]["S1"]
    a2 = load("runs/remote_a2/output_a2/summary.json")["results"]
    a3 = load("runs/remote_a3/output_a2_test/summary.json")["results"]
    a5 = load("runs/remote_a5/output_a5/summary.json")["results"]
    b1 = load("runs/remote_b1/output_b1/summary.json")["results"]
    a6 = load("runs/remote_a6/output_a6/summary.json")["results"]
    b2 = load("runs/remote_b2/output_b2/summary.json")["results"]
    R = []

    def site(res):
        d = res[MAIN]["decomposition"]
        return [("R1 ground conductivity, treated as address information, adds value: "
                 "$C(\\text{S0})-C(\\text{records})$, GBM, lower bound $>0$", iv(d["site_gain_gbm"]),
                 count(res, "R1_site")),
                ("R2 three questions from a 25-item pool recover most of the remaining value: $\\rho_3$ lower "
                 "bound $\\geq 0.5$", iv(d["rho3"]), count(res, "R2_short_questionnaire")),
                ("R3 a fixed order suffices: adaptive gain $D_\\text{static}$ upper bound $<0.02$",
                 iv(d["D_static"], 4), count(res, "R3_static_sufficient"))]

    def four(res):
        d = res[MAIN]["decomposition"]
        return [("R1 homeowner items add value: $C(\\text{T1--T2})-C(\\text{T1})$, GBM, lower bound $>0$",
                 iv(d["home_gain"]), count(res, "R1_home")),
                ("R2 three questions from homeowner and audit items recover most of their value: $\\rho_3$ "
                 "lower bound $\\geq 0.5$", iv(d["rho3"]), count(res, "R2_short_questionnaire")),
                ("R3 a fixed order suffices: $D_\\text{static}$ upper bound $<0.02$", iv(d["D_static"], 4),
                 count(res, "R3_static_sufficient")),
                ("R4 targeted ground testing works: $\\Phi_{0.2}$ lower bound $\\geq 0.5$ and above random",
                 iv(d["phi_evoi@0.2"]), count(res, "R4_targeted_test"))]

    def q(res):
        d = res[MAIN]["decomposition"]
        return [("Q1 homeowner items add value (network)", iv(d["nn_home_gain"]), count(res, "Q1_home_value")),
                ("Q2 homeowner gain exceeds audit gain added afterwards (network)", iv(d["home_minus_audit"]),
                 count(res, "Q2_home_over_audit")),
                ("Q3 three fixed homeowner questions recover most of the homeowner value: $\\rho_3$ lower bound "
                 "$\\geq 0.5$", iv(d["rho3_home_static"]), count(res, "Q3_short_home_questionnaire"))]

    R.append(("H-A1", "A tuning", [
        ("Adaptive questioning beats the best fixed order: $D_\\text{static}$ lower bound $\\geq 0.02$ "
         "(26-item pool including ground conductivity)", ivp(a1["D_static"], a1["D_static_ci"], 4),
         f"1/1 ({a1['decision'].replace('_', ' ').lower()})")],
              "Superseded: the adaptive gain came from ground conductivity in the question pool (H-A1S)."))
    R.append(("H-A1S", "A tuning", [
        ("As H-A1, with ground conductivity moved into the known information",
         ivp(a1s["D_static"], a1s["D_static_ci"], 4), f"1/1 ({a1s['decision'].replace('_', ' ').lower()})")],
              "Basis for treating conductivity as known; withdrawn after H-A4."))
    R.append(("H-A2", "A tuning", site(a2), "As H-A3."))
    R.append(("H-A3", "A test, first use", site(a3),
              "R1 held but its interpretation was withdrawn: H-A4 showed that ResStock samples conductivity per "
              "dwelling (PUMA explains 18.7\\% of its variance), so it is not address information. R2 and R3 "
              "assumed conductivity known and are not used. Led to the four-tier design."))
    R.append(("H-A4", "A tuning", [
        ("Diagnostic (no threshold): conductivity gain with all 22 packages and without the six GSHP packages",
         "0.165 [0.150, 0.183] vs. $-$0.002 [$-$0.005, 0.000]", "GSHP only")],
              "Reported in Section 5.4."))
    R.append(("H-A5", "A tuning", four(a5), "Replicated in H-B1."))
    R.append(("H-B1", "B test, first use", four(b1),
              f"R1, R2 replicated. R3 met only in the {SCN[[s for s in b1 if b1[s]['decomposition']['R3_static_sufficient']][0]]} "
              "scenario and was inconclusive in the other two. R4 failed in all scenarios: the ground test has "
              "almost no value in dataset B, so $\\Phi$ is a ratio of near-zero quantities. The paper's focus on "
              "homeowner questions was chosen after this result."))
    R.append(("H-A6", "A test, second use", q(a6) + [(r[0] + " (recomputed)", r[1], r[2]) for r in four(a6)],
              "Q1--Q3 are the claims of this paper."))
    r3b2 = [s for s in b2 if b2[s]["decomposition"]["R3_static_sufficient"]]
    R.append(("H-B2", "B test, second use", q(b2) + [(r[0] + " (recomputed)", r[1], r[2]) for r in four(b2)],
              "Q1--Q3 are the claims of this paper. In the recomputation R3 was met only in the "
              f"{', '.join(SCN[s] for s in r3b2)} scenario, not the one in H-B1: the R3 verdict in dataset B is "
              "not stable across re-runs of the same code, because $D_\\text{static}$ lies close to the 0.02 "
              "threshold."))
    return R


TEX = r"""\documentclass[preprint,11pt]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,tabularx,longtable}
\usepackage[margin=2.2cm]{geometry}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}
{\Large Supplementary Material}\\[0.6em]
{\large <<TITLE>>}\\[0.6em]
Jingjing Fan, Yunan Zhang
\end{center}

\section*{S1. Complete record of registered analyses}

Every analysis in this project was registered before it was run: the code was frozen with cryptographic
hashes, and the claims, thresholds and evaluation split were written down in a protocol. The two test splits
were each evaluated twice. Table~\ref{tabS1} lists every registered stage in order, the claims it tested, and
what became of them, including claims that were withdrawn or did not replicate. Estimates are for the main
emission scenario (Cambium long-run marginal rates, mid case) with 95\% PUMA cluster-bootstrap intervals; the
last column counts the emission scenarios in which the registered threshold was met (four in dataset A, three in
dataset B). Stages H-A1 and H-A1S used the value network; their thresholds applied to the main scenario only.

The analysis developed in three steps. (i)~H-A1 to H-A3 started from utility records plus ground thermal
conductivity, which was assumed to be obtainable from the address; on that basis the dataset A test split was
first used (H-A3). (ii)~A tuning-split diagnostic (H-A4) then showed that ResStock samples conductivity for each
dwelling and that its value comes entirely from ground-source heat pump (GSHP) packages, so it was moved into a
separate tier that requires a thermal response test. The four-tier decomposition was registered on the tuning
split (H-A5) and replicated on the dataset B test split (H-B1). (iii)~After H-B1, the paper was refocused on
homeowner questions, and the claims Q1--Q3 were registered and evaluated on both test splits (H-A6, H-B2).

Q1--Q3 are therefore not tests on untouched data. On the dataset B test split, the gradient-boosting homeowner
and audit gains and a short-questionnaire ratio over homeowner and audit items had already been seen (H-B1);
Q1--Q3 restate these comparisons for the value network and for homeowner items only. On the dataset A test split,
only the earlier tier definitions had been evaluated (H-A3). No model, setting or threshold was changed between the
evaluations of a split.

{\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{1.05cm}p{1.5cm}p{5.9cm}p{3.45cm}p{1.95cm}@{}}
\caption{Registered stages, claims and outcomes.}\label{tabS1}\\
\toprule
Stage & Split & Claim & Main scenario & Met \\
\midrule
\endfirsthead
\toprule
Stage & Split & Claim & Main scenario & Met \\
\midrule
\endhead
<<ROWS>>
\bottomrule
\end{longtable}}

\noindent Notation: $C$ is the recovery rate (0 = climate-zone default, 1 = per-dwelling optimum). S0 = utility
records plus ground conductivity. $\rho_3$ in H-A2/H-A3 is relative to the network with all 25 items answered,
starting from S0; in H-A5/H-B1 relative to the network with all 25 homeowner and audit items answered, starting
from records; for Q3 relative to all 15 homeowner items (Section 4.5). $D_\text{static}$ is the mean gain of
adaptive EVOI questioning over the fixed per-climate-zone order for one to four questions. $\Phi_{0.2}$ is the
share of the value of testing every dwelling that is recovered by testing the 20\% with the highest expected value
of testing. All protocols, frozen hashes and run summaries are in the code repository.

\end{document}
"""


def build(title):
    body = []
    for stage, split, claims, status in rows():
        for i, (c, e, m) in enumerate(claims):
            body.append(f"{stage if i == 0 else ''} & {split if i == 0 else ''} & {c} & {e} & {m} \\\\")
        body.append(r"\multicolumn{5}{@{}p{14cm}@{}}{\emph{Status:} " + status + r"} \\ \midrule")
    body[-1] = body[-1].replace(r" \\ \midrule", r" \\")
    tex = TEX.replace("<<TITLE>>", title).replace("<<ROWS>>", "\n".join(body))
    (OUT / "supplementary.tex").write_text(tex, encoding="utf-8")


TITLE = ("What should a homeowner be asked? Value of information for household-specific retrofit recommendation "
         "from two independent building-stock simulations")


if __name__ == "__main__":
    build(TITLE)
