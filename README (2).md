# DHSP-3W: Three-Way Decision for Selective Visual Question Answering

Reference implementation of the paper:

> **Three-Way Decision for Selective Visual Question Answering: Hedge Absorption Expands Coverage at Fixed Confident-Error Budget**
> Anup Batakurki, Ramesh Chundi
> School of Computer Science and Engineering, Dayananda Sagar University

---

## Table of contents

1. [Motivation](#motivation)
2. [Method in one paragraph](#method-in-one-paragraph)
3. [Key results](#key-results)
4. [What this repository is not](#what-this-repository-is-not)
5. [Installation](#installation)
6. [Data](#data)
7. [Quick start](#quick-start)
8. [Repository layout](#repository-layout)
9. [Reproducing paper tables](#reproducing-paper-tables)
10. [Reproducing paper figures](#reproducing-paper-figures)
11. [Configuration](#configuration)
12. [Evaluation protocol](#evaluation-protocol)
13. [Limitations](#limitations)
14. [Citation](#citation)
15. [Acknowledgments](#acknowledgments)
16. [Contact](#contact)
17. [License](#license)

---

## Motivation

Selective prediction for assistive VQA usually forces a binary answer-or-abstain decision under one error budget. That framing treats two error types as equivalent when they are not.

In VizWiz, a wrong answer from a VQA model can take one of two forms:

- The model outputs `unanswerable`. A blind user retries, rephrases, or asks someone else. This is a **hedged error (H-error)** and it is benign.
- The model outputs a confident wrong answer. The user may act on it — take the wrong pill, cross the wrong street. This is a **confident error (C-error)** and it is the safety-relevant failure mode.

Prior work counts both as "error" and reports a single budget `P(error) ≤ ε`. It also counts hedged outputs as answers, which inflates coverage. We correct both.

---

## Method in one paragraph

Given a calibrated C-error probability `p_C(e)` and a hedged-error probability `p_H(e)`, DHSP-3W applies a cost-sensitive argmax over three actions:

```
u_answer  = -λ_C · p_C(e) - λ_A · p_H(e)
u_hedge   = -λ_H
u_abstain = -λ_A
```

with `λ_A = 0.5` and `λ_H = 0.1`. The single tuned parameter is `λ_C`. For each budget `ε` we select the smallest `λ_C` such that the empirical C-error rate on a held-out calibration-fitting split does not exceed `ε`. Coverage counts only confident (non-hedged) answers.

---

## Key results

VizWiz 400-sample stratified test subset, coverage at matched confident-error budget:

| Method                | ε=0.02 | ε=0.05 | ε=0.10 | ε=0.15 | ε=0.20 |
|-----------------------|-------:|-------:|-------:|-------:|-------:|
| Token prob (mean)     | 0.050  | 0.087  | 0.205  | 0.237  | 0.295  |
| Token prob (min)      | 0.083  | 0.083  | 0.150  | 0.260  | 0.290  |
| Token prob (mean·min) | 0.065  | 0.065  | 0.163  | 0.235  | 0.295  |
| Head C (single)       | 0.018  | 0.018  | 0.018  | 0.240  | 0.285  |
| Ensemble C            | 0.003  | 0.003  | 0.030  | 0.258  | 0.280  |
| Meta head (binary)    | 0.172  | 0.172  | 0.172  | 0.275  | 0.345  |
| **DHSP-3W (ours)**    | 0.072  | 0.072  | **0.398** | **0.455** | **0.570** |

Paired bootstrap (2000 resamples) against token probability (mean):

| ε    | Δcoverage | 95% CI               | Verdict |
|------|-----------|----------------------|---------|
| 0.02 | +0.023    | [-0.008, +0.055]     | tie     |
| 0.05 | -0.015    | [-0.050, +0.020]     | tie     |
| 0.10 | **+0.192**| **[+0.153, +0.235]** | **win** |
| 0.15 | **+0.217**| **[+0.177, +0.260]** | **win** |
| 0.20 | **+0.276**| **[+0.230, +0.320]** | **win** |

Matched-coverage diagnostic (ablation against binary decision on the same meta head):

| Target coverage | 3-way C-error | Binary C-error |
|-----------------|--------------:|---------------:|
| 0.10            | 0.075         | 0.075          |
| 0.20            | 0.087         | 0.062          |
| 0.30            | 0.150         | 0.150          |
| 0.40            | 0.237         | 0.244          |
| 0.50            | 0.345         | 0.335          |

The two decision layers have statistically indistinguishable C-error rates at matched coverage. The coverage gain is therefore not from a better risk ranking. It comes from the hedge action absorbing H-errors that a binary policy would either answer or abstain on.

---

## What this repository is not

This is the **decision-layer reference implementation**, not a full assistive-VQA system. It expects pre-computed multimodal features from an upstream pipeline. In particular:

- We do not include the Qwen2.5-VL fine-tuning code. It is standard and out of scope.
- We do not include the OWLv2 / CLIP feature extractors. Use the upstream pipeline or any compatible feature cache.
- We do not claim to beat the single C-error head on AUROC. The meta head achieves AUROC 0.885, identical to the single head. Its value is combined scoring and calibration support.
- We do not claim the method wins at strict budgets. Binary decision wins at ε ≤ 0.05.

---

## Installation

Requires Python 3.10 or newer. Tested with PyTorch 2.1 on CUDA 12.1.

```bash
git clone https://github.com/your-username/dhsp-3w-selective-vqa.git
cd dhsp-3w-selective-vqa
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
# .venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

If you have a GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU-only runs are supported. Set `DEVICE=cpu` in the config or pass `--device cpu`.

---

## Data

The pipeline expects four pickle files produced by an upstream feature extraction step.

```
data/
├── risk_train_raw_cache.pkl       # 500 samples for head training
├── policy_train_raw_cache.pkl     # 100 samples (retained for compatibility)
├── calibration_raw_cache.pkl      # 500 samples for calibration
└── test_raw_cache.pkl             # 400 samples for evaluation
```

Each sample is a dict with at least the following keys:

| Key                  | Type       | Meaning |
|----------------------|------------|---------|
| `state_raw`          | list[11]   | Multimodal state: token-prob stats (Qwen2.5-VL), CLIP image-text similarity, OWLv2 detection confidence |
| `question`           | str        | VizWiz question text |
| `candidate_answer`   | str        | Model output (may be `"unanswerable"`, `"unsuitable"`, or empty) |
| `base_accuracy`      | float      | 1.0 if correct, 0.0 if incorrect |
| `unsafe_proxy_label` | int        | 1 for confident-wrong, 0 otherwise |

Optional: if `data/extra_safety_export/extra_safety_raw_cache.pkl` exists, its samples are appended to the calibration pool. This is the 72-sample safety supplement referenced in the paper.

We do not redistribute the cached features. Contact the authors for access, or reproduce them from the VizWiz benchmark with your own VLM pipeline.

---

## Quick start

### End-to-end via notebook

Open `notebooks/dhsp_3w_pipeline.ipynb` and run all cells. This is the recommended path on Kaggle and Colab.

### Command line

```bash
# Train all heads, calibrate, tune decision layer
python scripts/train_all.py --cache data/ --output runs/dhsp_3w/

# Evaluate against paper baselines and print tables
python scripts/evaluate.py --run runs/dhsp_3w/

# Regenerate all paper figures
python scripts/make_figures.py --run runs/dhsp_3w/ --out figures/
```

The three commands above reproduce every table and figure in the paper from a compatible feature cache.

---

## Repository layout

```
dhsp-3w-selective-vqa/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── dhsp_3w_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── risk_heads.py           # MLP training for base heads and ensembles
│   ├── meta_head.py            # stacked meta head over augmented features
│   ├── calibration.py          # temperature scaling + isotonic regression
│   ├── decision.py             # three-way cost-sensitive decision layer
│   ├── metrics.py              # coverage, C-error rate, class labels
│   └── evaluation.py           # paired bootstrap, Pareto sweep, ablation
├── scripts/
│   ├── train_all.py            # end-to-end training driver
│   ├── evaluate.py             # table generation
│   └── make_figures.py         # figure regeneration
├── data/
│   └── README.md               # pickle schema
├── figures/
│   ├── fig_coverage.png        # main result
│   ├── fig_matched_cov.png     # ablation diagnostic
│   ├── fig_reliability.png     # calibration
│   ├── fig_pareto.png          # utility frontier
│   └── fig_composition.png     # test-set class split
└── paper/
    ├── main.tex                # IEEEtran source
    ├── refs.bib
    └── README.md               # build instructions
```

---

## Reproducing paper tables

Every table in the paper maps to a function in `src/evaluation.py`. The mapping is:

| Paper table                  | Function / script               | Notebook cell |
|------------------------------|---------------------------------|---------------|
| Test composition             | `metrics.c_error_label` etc.    | Cell 4        |
| Table II — main results      | `evaluation.coverage_table`     | Cell 11       |
| Table III — bootstrap        | `evaluation.paired_bootstrap`   | Cell 14       |
| Table IV — isolation abl.    | `evaluation.isolation_ablation` | Cell 20       |
| Table V — matched coverage   | `evaluation.matched_coverage`   | Cell 21       |

Run the notebook cells in order, or run `scripts/evaluate.py` to produce the same tables as plain-text output.

---

## Reproducing paper figures

All figures in `figures/` are regenerated by `scripts/make_figures.py`:

| Figure                      | File                  | Notebook cell |
|-----------------------------|-----------------------|---------------|
| Coverage at matched budget  | `fig_coverage.png`    | Figure 1 cell |
| Matched-coverage diagnostic | `fig_matched_cov.png` | Figure 2 cell |
| Reliability diagram         | `fig_reliability.png` | Figure 3 cell |
| Pareto frontier (v2)        | `fig_pareto.png`      | Figure 4 cell |
| Error composition           | `fig_composition.png` | Figure 5 cell |

Each figure is generated standalone in the notebook so you can re-run any one without re-training the pipeline. The PNGs are 300 dpi and sized for a single IEEE column (3.5 in wide).

---

## Configuration

Key constants, all defined at the top of `notebooks/dhsp_3w_pipeline.ipynb` and mirrored in `src/decision.py`:

| Name           | Value | Meaning |
|----------------|-------|---------|
| `LAMBDA_A`     | 0.5   | Cost of abstaining |
| `LAMBDA_H`     | 0.1   | Cost of hedging |
| `LAMBDA_C`     | 10.0  | Default C-error cost; overridden by tuning per budget |
| `HARM_BUDGETS` | `[0.02, 0.05, 0.10, 0.15, 0.20]` | Sweep of ε |
| `COST_RATIOS`  | `[1, 2, 5, 10, 20, 50, 100]` | Utility sweep for Pareto |
| `N_BOOT`       | 2000  | Bootstrap resamples |
| `ENS_SEEDS`    | `[1, 7, 13, 21, 33]` | Ensemble seeds |

Change `LAMBDA_C` only via `tune_lambda_c`. The default value is used only when a budget-specific override is not available.

---

## Evaluation protocol

Every headline number in the paper is produced by the following protocol.

1. **Split discipline.** Heads and calibration are fit on `cal_fit` (285 samples). Thresholds are also tuned on `cal_fit`. `cal_conf` (287 samples) is used only for reliability reporting. The 400-sample test set is never seen during fitting.

2. **Confident-answer coverage.** A sample counts as covered only if the policy emits a non-hedged answer. Hedged outputs are treated as effective abstentions.

3. **C-error rate among answered queries.** The metric is `#C-errors / #answered`. H-errors are not counted. This is the safety-relevant metric.

4. **Budget tuning.** For each ε, the smallest λ_C is chosen such that the empirical C-error rate on `cal_fit` does not exceed ε. Coverage is monotone in ε across the observed range.

5. **Significance.** Paired bootstrap with 2000 resamples, percentile 95% CIs. Two-sided. Comparisons are done against the strongest baseline at each budget.

6. **Matched-coverage diagnostic.** At target coverage c, sort by risk ascending, walk down until confident-answer fraction reaches c. Report the C-error rate at that point. This is the ablation that isolates the decision layer from the risk score.

---

## Limitations

The paper discusses four limits. We restate them here so users do not over-read the numbers.

1. **Small safety-critical stratum.** The test subset contains only 8 samples matching a keyword-based safety criterion. Per-stratum results are diagnostic, not validated.

2. **Binary policy wins at strict budgets.** At ε ≤ 0.05, a binary decision on the same meta head wins by 10 points of coverage. The hedge term disqualifies samples the binary would safely answer.

3. **No finite-sample certification.** Clopper-Pearson upper bounds on the C-error rate exceed ε for ε ≤ 0.10 at n_cal = 287. Thresholds are empirically tuned, not certified.

4. **Meta head does not improve ranking.** AUROC against C-errors is 0.885, matching the single C-error head exactly. Its value is combined scoring and calibration, not a better ranker.

---

## Citation

If you use this code, please cite:

```bibtex
@article{batakurki2026dhsp3w,
  title   = {Three-Way Decision for Selective Visual Question Answering:
             Hedge Absorption Expands Coverage at Fixed Confident-Error Budget},
  author  = {Batakurki, Anup and Chundi, Ramesh},
  journal = {IEEE Transactions on Affective Computing (under review)},
  year    = {2026}
}
```

The upstream AERPN pipeline and the base feature extraction are described in:

```bibtex
@article{batakurki2025aerpn,
  title   = {AERPN: Adaptive Evidence-Based Risk-Aware Policy Network
             for Safe Assistive Visual Question Answering},
  author  = {Batakurki, Anup and Chundi, Ramesh},
  year    = {2025}
}
```

---

## Acknowledgments

The authors thank the School of Computer Science and Engineering, Dayananda Sagar University, for supporting this work. Experiments were run on Kaggle with NVIDIA Tesla T4 accelerators. The VizWiz benchmark is provided by the University of Texas at Austin.

---

## Contact

- **Anup Batakurki** — anupsb70@gmail.com
- **Ramesh Chundi** — chundiramesh@gmail.com

For bugs, open a GitHub issue. For collaboration, email both authors.

---

## License

MIT. See `LICENSE`. The VizWiz dataset is distributed under its own license and is not included in this repository. Users are responsible for complying with the VizWiz terms.
