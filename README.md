# G01 — Tesla asking-price regression (NYCU Data Mining HW1)

## Current report — 2026-10-19

`HW1_G01_report.pdf` is compiled from `latex/report.tex` and `latex/paper.tex` using the official ACL style. It has **six pages total, including references**, unchanged body font/margins, all seven members and the public repository URL on page one. The latest report is not a direct PDF overlay edit. The PDF, generated tables, source ZIP and reproducibility ZIP are synchronized.

Corrections: Section 4's H1 (trim flags), H2 (model-year interactions), and H3 (engineering versus raw ranking) are resolved in Table 7 using Table 3 and the B−C gap. Colour, country, model and power-squared checks are supplementary E1–E4 in Table 8. Table 6 includes VIF for all 28 terms. Section 6.2 and Table 4 consistently use 10,000 paired **dealer-cluster** bootstrap draws, seed 42. The three named Group B baselines are preserved. Related Work has a comparison table; member contributions are explicit. EUR 3,000 is an assumed tolerance without stakeholder cost evidence. No power EDA plot was supplied, so none was added.

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
| Table 4: dealer-cluster gaps | `analysis/comparisons.csv`; both Python commands |
| Table 5: model slices | `analysis/slices.csv`; base MAE recalculated; supplemental percentage summaries retained from team revision |
| Table 6: HC1 coefficients and VIF | `analysis/coefficients.csv`; both Python commands |
| Table 7: H1–H3 verdicts | computed from `analysis/results.csv` by `make_tables.py` |
| Table 8: supplementary E1–E4 | `analysis/team_extra_checks.csv`, transcribed from team PDF; `make_tables.py` formats it |
| Table 9: largest errors | `analysis/largest_errors.csv`; both Python commands |
| Figures 1/3: EDA/residuals | `src/recompute.py` |
| Figure 2: pipeline | `latex/paper.tex`; compile |

## Analysis and evidence boundaries

Set C is raw-EUR OLS with seven structured inputs, six title flags and three model-year interactions. KFold and dealer GroupKFold each have five folds; KFold seed 42. Transformations and ranking use fit partitions only. Test MAE remains EUR 4,289.57 (A), 3,711.48 (C), and 3,227.41 (RF). C and RF miss the assumed EUR 3,000 tolerance. Prior test exposure is disclosed; this is not a new blind evaluation.

HC1 robust t inference matches the team's revised methodology. Continuous/interaction terms are per SD, flag estimates/intervals in the displayed table are converted to 0-to-1 units. CSV retains the standardized fitting coefficients. VIF uses the full-rank design with an intercept. The saturated singleton Cybertruck remains in prediction, but its uncertainty is suppressed. Inference is exploratory and does not adjust for dealer dependence or multiple testing. Table 4 samples the 589 observed test dealer groups, keeping paired losses together and weighting by sampled row counts. It is conditional on fitted models, without retraining or selection uncertainty.

E1–E4, overlap descriptions and some narrative case/slice summaries are preserved from the supplied revised team PDF. Their original supplementary fold-level outputs were not supplied. `make_tables.py` reproduces their presentation from recorded values; **it does not rerun or independently verify their original paired tests**. See `analysis/PROVENANCE.md`. No invented p-values or new experiment outcomes are substituted.

## Packages and responsibilities

- `Tesla_G01_ACL_Source.zip`: current LaTeX, tables, figures, bibliography and official styles.
- `Tesla_G01_Reproducibility.zip`: current integration scripts/data/results/LaTeX, original `partB/`, requirements and README.
- A: 馮昱祥 515559007、林暉庭 314551160 — introduction, literature, data, EDA, feature pipeline.
- B: 杜洺鋒 315554001、邱育宏 315554053、吳宥駒 315551146 — model/tuning, validation, metrics/baselines and main comparisons.
- C: 黃少鋒 315561001、莊誌強 315561002 — ablation, coefficients/significance/VIF, errors and conclusion.

Earlier versions are recoverable in Git history. This update does not submit to E3 or the Judge. Review the scientific claims and prepare the separate presentation/submission package. ChatGPT/Codex assisted with source restoration, analysis verification, figures, writing and compilation; the team retains responsibility for verification.

## Wording and LaTeX correction

Section 8.1 uses the authors' voice and the supplied revised interpretation. H1–H3/E1–E4 identifiers, per-feature VIF and dealer-cluster bootstrap remain consistent. The source generator uses raw triple-quoted strings to preserve LaTeX commands and paired quotation marks in Cases 2, 3 and 5. This is a wording/typesetting update; no new experiment outcomes or reproducibility claims are added. The evidence boundaries above still apply.
