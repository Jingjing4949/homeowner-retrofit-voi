# What should a homeowner be asked?

Code, registered protocols and result summaries for

> J. Fan, Y. Zhang. *What should a homeowner be asked? Value of information for household-specific retrofit
> recommendation from two independent building-stock simulations.* Submitted to *Energy and Buildings*.

The study measures how much each kind of information (utility records, homeowner questions, auditor
measurements, a ground thermal response test) improves a household-specific retrofit recommendation, using
counterfactual simulations of every retrofit package in two NREL ResStock releases.

## Data

Both datasets are public and are downloaded anonymously from the NREL Open Energy Data Initiative (OEDI) S3 bucket:

| Dataset | Release | Download |
|---|---|---|
| A | ResStock 2025 Release 1 (22 packages) | `python h_a0.py fetch`, then `python h_a2.py fetch` (alternative emission scenarios) |
| B | ResStock 2024 Release 2, AMY2018 (16 packages) | `python h_b1.py fetch` |

The downloads are written to `cache/` and `cache_b1/` with a `MANIFEST.json` of file hashes.

## Registered analyses

Every stage was registered before it was run: the protocol (`protocols/`, in Chinese) fixes the claims,
thresholds and evaluation split, and `FREEZE.json` / `FREEZE_A0.json` record the SHA-256 hashes of the frozen
code, every revision and every execution. Supplementary Table S1 of the paper lists all stages and outcomes.

| Stage | Script | Split | Output |
|---|---|---|---|
| H-A0 | `h_a0.py run` | A tuning | (baseline tiers, gradient boosting) |
| H-A1 | `h_a1.py run` | A tuning | `runs/remote/output/` |
| H-A1S | `h_a1s.py run` | A tuning | `runs/remote_a1s/output_s/` |
| H-A2 | `h_a2.py run` | A tuning | `runs/remote_a2/output_a2/` |
| H-A3 | `h_a2.py test` | A test (first use) | `runs/remote_a3/output_a2_test/` |
| H-A4 | `h_a4.py run` | A tuning | `runs/remote_a4/output_a4/` |
| H-A5 | `h_a5.py run` | A tuning | `runs/remote_a5/output_a5/` |
| H-B1 | `h_b1.py run` | B test (first use) | `runs/remote_b1/output_b1/` |
| H-A6 | `h_a6.py A run` | A test (second use) | `runs/remote_a6/output_a6/` |
| H-B2 | `h_b2.py run` | B test (second use) | `runs/remote_b2/output_b2/` |

Each script also has a `smoke` stage that runs the pipeline on a small Colorado subset of the training split.
The claims of the paper (Q1–Q3) are those of H-A6 and H-B2.

`gshp_applicability.py` reproduces the GSHP applicability by ground conductivity level (Fig. 3a).
`paper/figures/make_figures.py` and `paper/latex/build_supp.py` regenerate the figures and Supplementary
Table S1 from the stored summaries.

## Environment

The registered runs used Python 3.11, PyTorch 2.14.1 (CUDA 13.0), LightGBM 4.7.0, NumPy 2.4.6, pandas 3.0.6 and
pyarrow 26.0.0 on a single NVIDIA RTX 5090 / RTX PRO 6000 GPU (see `requirements.txt`). Network training is not
bit-for-bit reproducible across GPUs; re-running H-B2 changed the network's three-question ratio over all 25
items from 0.78 to 0.73, while the registered thresholds were met with wide margins.

## License

Code: MIT (see `LICENSE`). ResStock data: see the NREL OEDI terms.
