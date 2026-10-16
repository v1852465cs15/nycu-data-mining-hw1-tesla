# G01 — Tesla asking-price regression (NYCU Data Mining HW1)

This integrated report has exactly six pages TOTAL, including references, using the official ACL style with unchanged margins and body font size. Member contributions follow the user's confirmed A/B/C assignments. The GitHub repository is private. Arrange instructor access before submitting the repository URL.

## Deliverables

- `HW1_G01_report.pdf`: preprint format, names/student IDs, repository URL on page one.
- `HW1_G01_judge.pdf`: review format, G01 only; no member names, student IDs or GitHub account in PDF text.
- `Tesla_G01_ACL_Source.zip`: editable LaTeX project, including both main documents, official styles, bibliography, generated tables and figures.
- `Tesla_G01_Reproducibility.zip`: calculation scripts, the unchanged shared workbook/settings, full results and LaTeX source.

No report has been submitted to E3 or the Judge. The required eight-minute presentation is a separate deliverable and is not included in this report package. The report is not a guarantee of any reviewer score.

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

Python 3.12 was used. Install `requirements.txt`, then run `python recompute.py` and `python make_tables.py` from the unpacked reproducibility folder. The optional local `deps` folder is not bundled; scripts otherwise use the installed Python environment. Model training reads `partB/Tesla.xlsx` and `partB/settings.yml` and writes only `analysis/` and generated tables/figures in `latex/`.

Workbook SHA-256: `92d5319a1f8781e8fbe0ec5d94a67285dead9bdd51da68a0e775deff83b70d3c`.

The benchmark preserves 3,002 training and 752 test rows. The final predictor is Set C OLS: seven structured fields, six title indicators and three model-by-registration-year interactions. Target is **raw asking price in EUR**, with no log transform. Both KFold and dealer GroupKFold have five folds; KFold seed is 42. Missing dealers are separate `_rowN` groups, matching the original code. Centering and scaling use fit partitions only. Top-k uses the actual standard error, not the fold standard deviation.

Test MAE: raw OLS 4,289.57 EUR; engineered OLS 3,711.48 EUR; random forest 3,227.41 EUR. Both engineered OLS and RF miss the proposed 3,000 EUR threshold. Bootstrap intervals are paired by test dealer cluster. HC3 coefficient inference excludes the saturated singleton Cybertruck row/dummy for the other coefficients; its own uncertainty is explicitly unavailable. The full model retains that row for prediction. Prior test exposure is disclosed; this is not a new blind evaluation.

`analysis/` includes final-model predictions, all coefficients, model slices, largest errors, bootstrap comparisons, k search, Ridge sensitivity, summary statistics and data-quality sensitivity. Existing team packages were inspected but no supplied source files were overwritten. References were checked against publisher/preprint materials.

## Remaining team actions

1. Review the integrated scientific claims before submission.
2. Arrange instructor access to this private repository so the code and data can be reviewed.
3. Prepare the matching presentation and the required E3 ZIP using the assignment naming rules.

## AI assistance

ChatGPT/Codex assisted with inspecting team material, correcting analysis code, rerunning experiments, creating plots, checking references, writing and compilation. Member attribution was confirmed by the user. The team retains responsibility for verifying the integrated scientific results.
