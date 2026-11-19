# Exact-Value Features for Fraud Detection on Synthetic Data

Code for an NTU AI6102 course project on [Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11): predict whether a card transaction is fraudulent, scored by ROC AUC. The 700k training rows are synthetic, inspired by a public 20k-row source table that is itself synthetic.

## Findings

- **Shallow trees are enough on the raw fields.** Depth-2 trees beat both stumps and deeper trees, and a logistic GLM on value encodings comes within 0.0007 AUC of the main model.
- **Value-level encodings add a gain on the competition data.** Per-value target encoding and decimal-digit features capture it. Real encodings beat encodings estimated from labels shuffled within 32 quantile bins by 0.0027 pooled out-of-fold AUC, averaged over three seeds, and all 15 seed–fold differences are positive. 0.0019 of this is a net gain over encoding only the narrow columns; the rest is what the shuffled encodings cost. A second fold split reproduces both gaps, and the gain is spread over five of the eight wide columns. It needs fine detail: the same encodings computed for 32 quantile bins give none of it (−0.0003 against the narrow-only model), 256 bins give +0.0003 and 2,048 bins +0.0009. A global label shuffle of the full pipeline scores 0.497.
- **The signal is local**, consistent with a fingerprint of the data generator. On the source table the AUC gain does not reproduce: the boosted model scores 0.9369 with these encodings and 0.9405 without them (a difference of −0.0037, 95% interval −0.0075 to +0.0002). The real-minus-shuffled direction keeps only about a quarter of its in-competition weight (transfer coefficient 0.22–0.28 vs 0.87–0.98), and about a third when the narrow-only model is the baseline (0.26 vs 0.78). Down-weighted to that level it no longer hurts there, but adds no detectable AUC.
- **Main model:** a GLM-initialised 4-leaf LightGBM. It reaches 0.95041 out-of-fold AUC (three-seed rank average) and 0.95062 on the public leaderboard, or 0.95186 when stacked with a public out-of-fold bank.

## Layout

```
src/        trainers, experiment grids, analyses and figures
codex/      ablation and cross-source scripts written by OpenAI Codex, with their small outputs
results/    logs and JSON files behind the numbers in the report
```

Data, model predictions and submissions are not included.

## Setup

The experiments ran with Python 3.12 on Linux. The XGBoost and CatBoost runs in grids v1–v2 use a CUDA GPU; everything else runs on the CPU.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`src/run_grid.sh` uses GNU `xargs -d`, so run the grids on Linux.

## Data

Download the competition data with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) after accepting the competition rules. Keep the zip as well, because the scripts in `codex/` read it directly.

```bash
kaggle competitions download -c playground-series-s6e11 -p data
unzip data/playground-series-s6e11.zip -d data
```

Save the source table, [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) (CC0), as `data/credit_card_fraud_2026.csv`.

Only the stacking and the bonus blend need these public inputs:

- [S6E11 OOF Bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank) (CC0), unpacked into `public/oof_bank/`.
- `submission.csv` from version 7 of the notebook [S6E11 LM Ladder | Forecast the Board](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-forecast-the-board-lb-0-95417) (Apache 2.0), saved as `public/lm_ladder/lm_ladder_submission_v7.csv`.

## Reproducing the report

Run everything from the repository root. Each line of a grid file is one 5-fold CV run on the same outer folds (fold seed 42). A run writes its out-of-fold and test predictions to `runs/<name>/` and appends a line to `runs/log.jsonl`. `src/summarize.py` prints that log as a table. Each script's docstring lists its arguments.

| Report item | Commands |
|---|---|
| Model family and interaction order, effect of tree depth, features tried and dropped | `bash src/run_grid.sh src/grid_v1.txt 7` |
| Feature ladder, leakage and smoothing checks | `bash src/run_grid.sh src/grid_v2.txt 7` |
| Exact-value encodings of three preset fields (4-leaf model; an early check that the report no longer tabulates), main-model seeds | `bash src/run_grid.sh src/grid_v3.txt 7` |
| Global label shuffle, within-bin label shuffling and the narrow-only arm (strict protocol) | grids `v4` and `v6`, then `src/placebo_compare.py` and `src/narrow_compare.py` |
| Where the gain sits (one wide column at a time) and the second fold split | `bash src/run_grid.sh src/grid_v7.txt 3` (writes to `runs/v7/`; the `fs7_*` runs use fold seed 7), then `src/column_compare.py` |
| Fraud rates for 32, 256 or 2,048 bins instead of exact values | `bash src/run_grid.sh src/grid_v9.txt 3` (uses `--te_bins`; writes to `runs/v9/`), then `src/bins_compare.py` |
| Binned additive logit with and without preset interactions | `codex/interaction_probe.py data/playground-series-s6e11.zip` |
| Exact-value encodings of three preset fields (full model; lower block of the per-column table) | `codex/full_model/run_fold.py --fold K` for K = 1..5, then `codex/full_model/score.py` |
| Public-board noise | `src/lb_noise.py`; the report figure comes from `src/figures_report.py` |
| Ensembles and submissions | `src/ensemble.py`, `src/stack_bank.py` |
| Transfer test on the source table | `src/xsource_precheck.py`, grid `v5` (`src/xsource_train.py`), `src/xsource_eval.py`, `src/xsource_beta.py`, `src/xsource_shrink.py` (down-weighting check), grid `v8` (narrow-only arm; then `src/xsource_eval.py` with that run in place of the shuffled one), `codex/source_transfer/score_transfer.py` |
| Report figures | `src/figure_gain.py` (the main-result figure: change in AUC against the narrow-only model on both tables, read from `results/bins_compare.json` and `results/eval_boost_narrow_s42.json`); `src/figures_report.py` (public-board noise and transfer weights; it also draws a per-fold depth plot that the report no longer includes) |

`codex/value_probe.py data/playground-series-s6e11.zip` is a univariate check that compares exact-value, binned and within-bin-shuffled encodings of the three preset fields. Its outputs are `codex/value_*.csv`.

## Notes

- `src/xsource_train.py` is `src/train2.py` with `codex/source_transfer/transfer.patch` applied. It also scores the 20k source rows.
- `codex/` was called `codex_review/` when the runs were made, so paths recorded in `results/*.json` still use the old name.
- File names, flags and logs keep earlier names: `placebo` (`--te_placebo 1`) is the report's within-bin label shuffling, and `SHUFFLED` (`--shuffle_labels 1`) is the global label shuffle.
- Per-value target encoding, digit keys, shallow trees and GLM-initialised boosting come from the Kaggle community. [Discussion 755149](https://www.kaggle.com/competitions/playground-series-s6e11/discussion/755149) collects the sources.
- The code was written with AI coding assistants (Claude Code and OpenAI Codex) under the author's direction. Codex wrote the scripts in `codex/`; the report's AI statement gives details.

## License

MIT, see [LICENSE](LICENSE).
