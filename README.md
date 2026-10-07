# Split and Leaf Hijacking: code, seeds and results

Code accompanying the paper *Split and Leaf Hijacking: Integrity Attacks on Federated Gradient-Boosted Trees* (Turke Althobaiti, Habib Ullah Manzoor, Basim Alhumaily, Naeem Ramzan).

## About this research

**The setting.** Federated gradient-boosted trees let several organizations (banks, hospitals, ...) train one tree ensemble without sharing raw data. Two protocols are in use: horizontal FL (HFL), where each client holds different *rows* and the server sums every client's gradient histogram to pick each split, and vertical FL (VFL/SecureBoost), where each party holds different *columns* and a passive party builds an encrypted histogram under Paillier homomorphic encryption. Either way, every node of every tree is decided by a **discrete argmax** over an aggregated statistic.

**The gap.** Most federated learning security research targets neural networks trained with FedAvg-style continuous averaging, and most federated-tree security research targets privacy (reconstructing data, labels or splits from what is shared), not integrity. Nobody had asked what happens when the histogram a client reports, or the routing decision a passive party returns, is simply false. A discrete argmax behaves differently from a continuous average: the attacker's goal is not to shift a weighted mean a little, but to push one specific candidate split past the current winner.

**The idea.** We treat this as a margin problem. For HFL we derive a closed-form **split flip margin**: the smallest forged amount that one histogram cell needs so that a chosen candidate split overtakes the current winner. This margin is provably invariant to how rows are partitioned across clients (for fixed bin edges), and we prove that, when reports are unrestricted, **one malicious client can reach anything that several colluding clients could reach** — so raw attacker count stops mattering until per-client reports are bounded.

**The attacks.**
- **Split hijacking** — a client (HFL) forges its histogram, or a passive party (VFL) rescales its encrypted gradient contribution, so that a different split is chosen at a node. This changes *what the tree learns*.
- **Leaf hijacking** — after an honest split is chosen, a passive party returns false left/right routing decisions for a chosen fraction of samples. This changes *which leaf value a sample ends up with*, without touching split selection at all.
- A **label-free prediction-disruption mechanism**: a VFL passive party can plant a split on its own feature and, at inference time, move a chosen input across that threshold to flip its prediction — without ever seeing a label.

**What we found.** A single attacker at a modest, bounded reach (one node and its two children) causes only a small AUC drop on the datasets we test; the effect grows sharply with how much of the tree the attacker can corrupt, and full-tree coverage can collapse a model to a constant predictor. An attacker who sees nothing but its own report still does real damage, just less of it. A per-client bound on report size (a simple, practical defense) restores the usual "you need several colluders" property that unrestricted reporting removes. We also test whether simple integrity checks (conservation, magnitude, norm) catch these attacks, and show a variant that is tuned to pass a conservation check while keeping most of its effect. Full numbers, confidence intervals and the scope of every claim are in the paper (`revision/paper/paper.tex`) and the reproduction instructions below.

## Layout

| Path | Content |
|---|---|
| `src/harness/` | Our protocol-faithful implementations: horizontal histogram-summing GBDT (`federated_gbdt.py`) and Paillier-based vertical SecureBoost (`vfl_secureboost.py`). |
| `src/attacks/` | Split hijacking (`histogram_integrity.py`, `he_crypto_layer.py`), leaf hijacking (`leaf_routing.py`), the combined attack (`multi_vector.py`), prediction disruption (`passive_party_backdoor.py`), target selection. |
| `src/theory/margin.py` | Closed-form split flip margin. |
| `src/fed_datasets/` | Loaders for UCI Adult, Heart Disease (Cleveland) and Default of Credit Card Clients. |
| `experiments/` | Experiments of the original submission (kept for reference; not used for the revised numbers). |
| `revision/experiments/` | Experiments behind every number, figure and table of the revised paper, and the scripts that write those numbers into the manuscript text (`build_results.py`, `build_discussion.py`, `build_letter.py`) and check it (`numbering.py`, `revfloat.py`, `check_paper.py`, `story_check.py`). |
| `revision/results/` | JSON results, raw logs and the per-task VFL cache. |
| `revision/paper/` | The manuscript (`paper.tex`, with every revision change marked `\rev{...}`; `paper_clean.tex` without the markup; `ref.bib`), so `build_*.py` above can regenerate it end to end. |
| `revision/figures/` | Every figure used by the revised paper, as PDF and PNG (regenerate with `make_figures.py` / `make_figures_u.py`). |
| `revision/response_to_reviewers.tex` | The point-by-point response to the reviewers, generated by `build_letter.py` from the same result files. |
| `results/original_*` | Cached results and figures of the original submission. |

## Setup

Python 3.11. Tested with numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.8, phe 1.5.0, xgboost 3.2.0 (only for the honest-training check).

```
pip install -r requirements.txt
```

Download the three UCI datasets (Adult, Heart Disease/Cleveland, Default of Credit Card Clients) from the UCI Machine Learning Repository into `data/raw/adult/`, `data/raw/heart_disease/` and `data/raw/credit_default/`. The loaders in `src/fed_datasets/loaders.py` list the exact file names they read. We do not redistribute the data.

## Shared protocol

Every revised experiment uses `revision/experiments/unified.py`. For each seed it subsamples the data where configured, draws a stratified train-test split with `random_state=seed`, and (HFL) a Dirichlet(0.5) partition over five clients with the same seed. A clean run and every attacked run of the same dataset and seed share all draws, so each effect is a paired difference with a 95% t-interval. HFL experiments use seeds 0-9. VFL experiments use seeds 0-4, except the Adult headline experiments (scale, leaf hijacking, prediction disruption, target selection), which use seeds 0-9. Configuration per dataset (bins, depth, rounds) is in `unified.py` and in the configuration table of Section 5.2.

## Reproducing the revised results

Run from `revision/experiments/`, one script at a time. The Paillier experiments are slow (about 1.5 minutes per training at 20 rounds), so `exp_u_vfl.py` is resumable: it writes one JSON per task into `revision/results/vfl_u_cache/`.

```
python exp_hfl_all.py            # attacker variants, bounds, persistence at the default radius (ATTACK_DEPTH=2)
python exp_u_hfl.py              # attacker count, radius, federation size, transient rounds, depth, aggregation controls
python exp_u_extra.py            # released-tree attacker and budget-matched baselines
python exp_fedbins.py            # federated binning
python exp_xgb_conformance.py    # honest-training check against XGBoost
python check_constant.py         # constant-predictor check at full radius
python exp_u_vfl.py              # VFL experiments at 20 rounds (use VFL_EXT=1 for the Adult seeds 5-9)
python exp_trigger_specificity.py  # false-trigger baseline of the prediction-disruption manipulation
python resolve_overflow.py       # see "Known limitation" below
python excess_flip.py && python aggregate_vfl.py
python make_figures.py && python make_figures_u.py
python build_results.py && python build_discussion.py && python build_letter.py
```

This full pipeline is self-contained in this repository, including the last step: `build_results.py` and `build_discussion.py` rewrite `revision/paper/paper.tex` from the JSON files above, and `build_letter.py` rewrites `revision/response_to_reviewers.tex` (and embeds the four new figures from `revision/figures/`), so every number, figure and sentence that states a number can be regenerated from scratch and traced back to the script that produced it. `revision/paper/check_paper.py` and `revision/paper/story_check.py` (run with `revision/paper/` as the working directory) check the regenerated manuscript for structural errors and leftover revision-history wording.

Earlier runs that the final results supersede are kept for reference: `exp_matched_core.py` and `results/matched_core*` (replaced by `exp_u_hfl.py`), `results/hfl_all_rootonly.*` (root-only radius), `exp_vfl_converged.py` runs and `results/vfl_summary.json` (replaced by `exp_u_vfl.py` and `results/u_vfl_summary.json`; `exp_vfl_converged.py` is still imported for the target-selection task).

## Radius convention

Depth is indexed from zero. Radius `r` corrupts nodes of depth at most `r`; the default `r=1` corrupts the root and its two children. In code this is `attack_depth = r + 1`.

## Which attack code produced which numbers

All HFL numbers of the revised paper come from `revision/experiments/rev_attacks.py`, which selects the target candidate with the per-feature totals that the server's split search and `src/theory/margin.py` use. The attack in `src/attacks/histogram_integrity.py` (`margin_flip_attack`), used by the scripts in `experiments/` for the original submission, chooses its target with totals over the whole histogram. That convention is kept only so the original results can be reproduced. The two give the same default-radius loss on Adult (0.0087 and 0.0088) but are not the same target-selection rule.

## Scope

All attack results apply to the implementations in this repository. FedTree v1.0.5 was built from source and its bundled examples were run as an honest-baseline check only. The attacks were not run against FedTree or any other named production system. The honest HFL model was compared with XGBoost's histogram method (paper, Section 5.7). XGBoost reaches a higher test AUC on all three datasets, and we did not identify the cause.

## Known limitation

Four VFL tasks raised a Paillier `OverflowError` on first execution: one in an early batch (task not identified) and three in the final batch (`bd_credit_8_1`, `leaf_adult_1.0_7`, `party_3_0`). Each was rerun three times with fresh keys, all reruns succeeded with identical results, and the rerun values are the ones used. `revision/results/overflow_events.json` records the events. We cannot explain them.

## Cite this paper

```bibtex
@article{althobaiti2026splitleaf,
  title   = {Split and Leaf Hijacking: Integrity Attacks on Federated Gradient-Boosted Trees},
  author  = {Althobaiti, Turke and Manzoor, Habib Ullah and Alhumaily, Basim and Ramzan, Naeem},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License (see `LICENSE`).
