# G01 — Tesla asking-price regression (NYCU Data Mining HW1)

## Current report — 2026-10-19

`HW1_G01_report.pdf` is compiled from `latex/report.tex` and `latex/paper.tex` using the official ACL style. It has **six pages total, including references**, unchanged body font/margins, all seven members and the public repository URL on page one. The latest report is not a direct PDF overlay edit. The PDF, generated tables, source ZIP and reproducibility ZIP are synchronized.

Corrections: Section 4's H1 (trim flags), H2 (model-year interactions), and H3 (engineering versus raw ranking) are resolved in Table 7 using Table 3 and the B−C gap. Colour, country, model and power-squared checks are supplementary E1–E4 in Table 8. Table 6 includes VIF for all 28 terms. Table 4 uses 10,000 paired **car-level** bootstrap draws (seed 42), matching the requested comparison image. Section 6.2 distinguishes this table from the dealer-cluster sensitivity retained in `analysis/comparisons.csv`. The three named Group B baselines are preserved. Related Work has a comparison table; member contributions are explicit. EUR 3,000 is an assumed tolerance without stakeholder cost evidence. No power EDA plot was supplied, so none was added.

## Data and original Group B implementation

AutoScout24 European Tesla advertisements collected **2026-10-05**: 3,754 records, 3,002 training and 752 test. Target: asking price in EUR, not completed sale price. The supplied split is retained. Workbook SHA-256: `92d5319a1f8781e8fbe0ec5d94a67285dead9bdd51da68a0e775deff83b70d3c`.

Original Group B files are in [`partB/`](partB/), copied byte-for-byte from `partB (1).zip`, preserving separate modules, indentation, fonts and settings. Follow its README from inside that directory; run `models.py` before analysis. `clear.py` deletes local Trackio projects and is not needed to reproduce this report. The original files are unchanged by this revision. Other team-uploaded files under `src/` are retained; the integration entry points are specifically `src/recompute.py` and `src/make_tables.py`.

## Reproduce and compile

Python 3.12; install `requirements.txt`. Run from the repository root:

```sh
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/recompute.py
python src/make_tables.py
cd latex
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

The output is `latex/report.pdf`; copy it to the repository root as `HW1_G01_report.pdf` after review. To compile only, use the saved `latex/` files and the last four commands. Overleaf: upload the source ZIP, choose **XeLaTeX** and main file `report.tex`. Times New Roman/MingLiU are used locally; TeX Gyre Termes/FandolSong are fallbacks, so hosted font differences can alter pagination.

| Report item | Source and reproduction |
|---|---|
| Related Work comparison | `latex/paper.tex`, `latex/references.bib`; compile |
| Tables 1–3: summary, results, ablation | `analysis/summary.json`, `results.csv`; both Python commands |
| Table 4: car-level paired gaps | `analysis/car_comparisons.csv`; both Python commands; dealer-cluster sensitivity remains in `comparisons.csv` |
| Table 5: model slices, Gain and All | `analysis/slice_metrics.csv`; all columns computed from matching test predictions by both Python commands |
| Table 6: HC1 coefficients and VIF | `analysis/coefficients.csv`; both Python commands |
| Table 7: H1–H3 delta and paired p-values | `analysis/fold_losses.csv`, `hypothesis_tests.csv`; both Python commands |
| Table 8: supplementary E1–E4 | `analysis/team_extra_checks.csv`, transcribed from team PDF; `make_tables.py` formats it |
| Table 9: largest errors | `analysis/largest_errors.csv`; both Python commands |
| Figures 1/3: EDA/residuals | `src/recompute.py` |
| Figure 2: pipeline | `latex/paper.tex`; compile |

## Analysis and evidence boundaries

Set C is raw-EUR OLS with seven structured inputs, six title flags and three model-year interactions. KFold and dealer GroupKFold each have five folds; KFold seed 42. Transformations and ranking use fit partitions only. Test MAE remains EUR 4,289.57 (A), 3,711.48 (C), and 3,227.41 (RF). C and RF miss the assumed EUR 3,000 tolerance. Prior test exposure is disclosed; this is not a new blind evaluation.

HC1 robust t inference matches the team's revised methodology. Continuous/interaction terms are per SD, flag estimates/intervals in the displayed table are converted to 0-to-1 units. CSV retains the standardized fitting coefficients. VIF uses the full-rank design with an intercept. The saturated singleton Cybertruck remains in prediction, but its uncertainty is suppressed. Inference is exploratory and does not adjust for dealer dependence or multiple testing. Table 4 uses a common matrix of 10,000 car-level index draws for all three paired comparisons. The dealer-cluster sensitivity separately samples 589 observed test dealer groups, keeping paired losses together and weighting by sampled row counts. It is conditional on fitted models, without retraining or selection uncertainty.

E1–E4, overlap descriptions and some narrative case summaries are preserved from the supplied revised team PDF. Their original supplementary fold-level outputs were not supplied. `make_tables.py` reproduces their presentation from recorded values; **it does not rerun or independently verify their original paired tests**. See `analysis/PROVENANCE.md`. No invented p-values or new experiment outcomes are substituted.

## Packages and responsibilities

- `Tesla_G01_ACL_Source.zip`: current LaTeX, tables, figures, bibliography and official styles.
- `Tesla_G01_Reproducibility.zip`: current integration scripts/data/results/LaTeX, original `partB/`, requirements and README.
- A: 馮昱祥 515559007、林暉庭 314551160 — introduction, literature, data, EDA, feature pipeline.
- B: 杜洺鋒 315554001、邱育宏 315554053、吳宥駒 315551146 — model/tuning, validation, metrics/baselines and main comparisons.
- C: 黃少鋒 315561001、莊誌強 315561002 — ablation, coefficients/significance/VIF, errors and conclusion.

Earlier versions are recoverable in Git history. This update does not submit to E3 or the Judge. Review the scientific claims and prepare the separate presentation/submission package. ChatGPT/Codex assisted with source restoration, analysis verification, figures, writing and compilation; the team retains responsibility for verification.

## Wording and LaTeX correction

Section 8.1 uses the authors' voice and the supplied revised interpretation. H1–H3/E1–E4 identifiers and per-feature VIF are retained; Table 4 now follows the requested car-level version. The source generator uses raw triple-quoted strings to preserve LaTeX commands and paired quotation marks in Cases 2, 3 and 5. This is a wording/typesetting update; no new experiment outcomes or reproducibility claims are added. The evidence boundaries above still apply.

## Restored table details and individual contributions

Table 7 contains recomputed delta (p) for KFold and GroupKFold: flags 213 (.0003) / 198 (.002); interactions 462 (.0001) / 455 (.004); B−C 723 (.0001) / 698 (.002). `fold_losses.csv` saves all matched fold MAEs; `hypothesis_tests.csv` saves full-precision deltas and two-sided paired t-test values. Five overlapping training fits mean these tests remain exploratory.

Table 5 includes Gain = 100(C/A−1)% and All across 752 test cars. All MAE, gain, median absolute percentage error and within-EUR3000 shares are calculated from the predictions, not copied from a report. Table 4 now displays [417,742], [453,783], [-723,-236], reproduced with car-level paired draws. Its p column still uses the grouped-fold paired t-test; it is not a bootstrap p-value.

Section 7.1 includes the 600 non-overlapping-row sensitivity: RF's advantage over C is EUR215, its excess over the threshold is EUR816, and C's advantage over A is EUR519. Contributions have one entry for each of the seven members, with concrete shared tasks and section/figure references; no unsupported individual division of work is invented. The report remains six pages including references.

馮昱祥 (515559007) additionally organized the report and formatted it using the ACL template; this responsibility is included in the individual Contributions entry.
