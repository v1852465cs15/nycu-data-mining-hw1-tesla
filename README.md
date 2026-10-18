# G01 — Tesla asking-price regression (NYCU Data Mining HW1)

This integrated report has exactly six pages TOTAL, including references, using the official ACL style with unchanged margins and body font size. Member contributions follow the user's confirmed A/B/C assignments. The GitHub repository is public and can be viewed without signing in.

## Quick start and repository layout

Source: AutoScout24 European Tesla advertisements collected **2026-10-05**. The target is advertised asking price in EUR, not a completed sale price. The supplied workbook has **3,754 records: 3,002 training and 752 test**; the existing split is retained.

- `data/Tesla.xlsx`: shared dataset, unchanged; `data/settings.yml`: worksheet and feature settings.
- `src/recompute.py`: preprocessing, feature construction, training, validation, evaluation and figures.
- `src/make_tables.py`: converts saved analysis results into LaTeX tables.
- `analysis/`: saved numerical outputs used by the report.
- `latex/`: editable ACL source and figures; compile `report.tex`.
- `HW1_G01_report.pdf`: current six-page formal report. ZIPs are retained for download.

Python 3.12 installation and execution, from the repository root:

```sh
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/recompute.py
python src/make_tables.py
cd latex
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```

The last four commands require XeLaTeX/BibTeX (MiKTeX or TeX Live). Alternatively upload `latex/` to Overleaf and select XeLaTeX.

### Reproducing report tables and figures

| Report item | Saved result / source | Reproduction |
|---|---|---|
| Table 2, main comparison | `analysis/results.csv`, `latex/results_table.tex` | Both Python commands above |
| Table 3, ablation | `analysis/results.csv`, `latex/ablation_table.tex` | Both Python commands above |
| Table 4, paired gaps | `analysis/comparisons.csv`, `latex/comparisons_table.tex` | Both Python commands above |
| Figure 1, EDA | `latex/figures/eda.pdf` | `python src/recompute.py` |
| Figure 2, pipeline | Embedded diagram in `latex/paper.tex` | Compile `report.tex` |
| Figure 3, residuals | `latex/figures/residuals.pdf` | `python src/recompute.py` |

This repository reorganization changes paths only; no new experiments or numerical changes were made. The 2026-10-18 formal PDF contains only the requested baseline-label wording update. Part C's later revision list is not applied. The earlier anonymous report is removed from the current branch, but remains recoverable in Git history.

## Deliverables

- `HW1_G01_report.pdf`: preprint format, names/student IDs, repository URL on page one.
- `Tesla_G01_ACL_Source.zip`: editable LaTeX project, including both main documents, official styles, bibliography, generated tables and figures.
- `Tesla_G01_Reproducibility.zip`: calculation scripts, the unchanged shared workbook/settings, full results and LaTeX source.

This update does not submit a report to E3 or the Judge. The required eight-minute presentation is a separate deliverable and is not included in this report package. The report is not a guarantee of any reviewer score.

## Compile

Use **XeLaTeX**, then BibTeX, then XeLaTeX twice. In Overleaf select XeLaTeX and main document `report.tex` or `judge.tex`. The local build uses Times New Roman and MingLiU; TeX Gyre Termes and FandolSong-Regular are fallbacks for hosted TeX installations. Do not compile this source using pdfLaTeX.

```
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```

Edit `team.tex` for team/repository information and `paper.tex` for any future changes to contributions. Confirmed assignments: A — 馮昱祥 515559007, 林暉庭 314551160; B — 邱育宏 315554053, 吳宥駒 315551146, 杜洺鋒 315554001; C — 黃少鋒 315561001, 莊誌強 315561002. Responsibilities are shared within each subgroup, as specified by the user. The anonymous version replaces member identities with member numbers.

## Reproduce

Python 3.12 was used. Install `requirements.txt`, then run `python recompute.py` and `python make_tables.py` from the unpacked reproducibility folder (legacy ZIP layout); in this repository use the `src/` commands above. The optional local `deps` folder is not bundled; scripts otherwise use the installed Python environment. Model training reads `partB/Tesla.xlsx` and `partB/settings.yml` and writes only `analysis/` and generated tables/figures in `latex/`.

Workbook SHA-256: `92d5319a1f8781e8fbe0ec5d94a67285dead9bdd51da68a0e775deff83b70d3c`.

The benchmark preserves 3,002 training and 752 test rows. The final predictor is Set C OLS: seven structured fields, six title indicators and three model-by-registration-year interactions. Target is **raw asking price in EUR**, with no log transform. Both KFold and dealer GroupKFold have five folds; KFold seed is 42. Missing dealers are separate `_rowN` groups, matching the original code. Centering and scaling use fit partitions only. Top-k uses the actual standard error, not the fold standard deviation.

Test MAE: raw OLS 4,289.57 EUR; engineered OLS 3,711.48 EUR; random forest 3,227.41 EUR. Both engineered OLS and RF miss the proposed 3,000 EUR threshold. Bootstrap intervals are paired by test dealer cluster. HC3 coefficient inference excludes the saturated singleton Cybertruck row/dummy for the other coefficients; its own uncertainty is explicitly unavailable. The full model retains that row for prediction. Prior test exposure is disclosed; this is not a new blind evaluation.

`analysis/` includes final-model predictions, all coefficients, model slices, largest errors, bootstrap comparisons, k search, Ridge sensitivity, summary statistics and data-quality sensitivity. Existing team packages were inspected but no supplied source files were overwritten. References were checked against publisher/preprint materials.

## Remaining team actions

1. Review the integrated scientific claims before submission.
2. Confirm that the final report and repository agree before submission.
3. Prepare the matching presentation and the required E3 ZIP using the assignment naming rules.

## AI assistance

ChatGPT/Codex assisted with inspecting team material, correcting analysis code, rerunning experiments, creating plots, checking references, writing and compilation. Member attribution was confirmed by the user. The team retains responsibility for verifying the integrated scientific results.

## 2026-10-18 text revision

The formal report names and bolds Trivial Baseline, Simple Baseline and Strong Baseline as requested by Group B. Numerical results and analyses are unchanged. Float spacing was adjusted to retain six total pages. The Part C revision list has not been applied. The earlier judge PDF is available only in Git history. The source archives contain the revised shared paper text.
