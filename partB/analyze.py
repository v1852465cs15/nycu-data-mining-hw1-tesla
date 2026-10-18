import io
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager

font_manager.fontManager.addfont("fonts/NotoSansTC-Regular.otf")
plt.rcParams["font.sans-serif"] = ["Noto Sans TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import trackio
from PIL import Image
from scipy import stats

from data import settings, test_df, y_test
from models import MODELS, OLS_MODELS, model_path, report_path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = settings["trackio"]["project"]

if settings["trackio"]["clear-on-analyze"]:
    trackio.delete_project(PROJECT, force=True)


def r2_ylim(ax, values):
    """R2 axis from below zero to 1, so negative R2 (worse than the mean) is visible."""
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylim(min(-0.25, np.min(values) - 0.1), 1.0)


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return trackio.Image(Image.open(buf))


# ==========================================
# Shared Analysis (every model)
# ==========================================


def log_common(name, report, y_true, y_pred):
    residuals = y_true - y_pred
    pct_err = np.abs(residuals) / y_true * 100

    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    test_r2 = r2_score(y_true, y_pred)

    cv_mae, cv_rmse, cv_r2 = (np.array(report[k]) for k in ("cv_mae", "cv_rmse", "cv_r2"))

    # Summary metrics
    trackio.log(
        {
            "summary/test_mae": test_mae,
            "summary/test_rmse": test_rmse,
            "summary/test_r2": test_r2,
            "summary/cv_mae_mean": cv_mae.mean(),
            "summary/cv_rmse_mean": cv_rmse.mean(),
            "summary/cv_r2_mean": cv_r2.mean(),
            "summary/median_pct_error": np.median(pct_err),
        }
    )

    # GroupKFold summary (only in reports trained after GroupKFold was added)
    if "group_cv_mae" in report:
        trackio.log(
            {
                "summary/group_cv_mae_mean": np.mean(report["group_cv_mae"]),
                "summary/group_cv_rmse_mean": np.mean(report["group_cv_rmse"]),
                "summary/group_cv_r2_mean": np.mean(report["group_cv_r2"]),
            }
        )

    # Cross-validation folds (one step per fold)
    for fold in range(len(cv_rmse)):
        trackio.log(
            {"cv/mae": cv_mae[fold], "cv/rmse": cv_rmse[fold], "cv/r2": cv_r2[fold]},
            step=fold + 1,
        )

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect Prediction")
    ax.set(
        xlabel="Actual Price",
        ylabel="Predicted Price",
        title=f"{name}: Actual vs Predicted",
    )
    ax.legend()
    ax.grid(True)

    trackio.log(
        {
            "plots/actual_vs_predicted": to_image(fig),
            "dist/residuals": trackio.Histogram(residuals, num_bins=40),
            "dist/pct_error": trackio.Histogram(np.clip(pct_err, 0, 100), num_bins=40),
        }
    )

    print(f"{name}: CV RMSE {cv_rmse.mean():.2f}" f" | Test MAE {test_mae:.2f}, RMSE {test_rmse:.2f}, R2 {test_r2:.4f}")
    return cv_r2, test_r2


# ==========================================
# Main Model Analysis
# ==========================================


def log_main(name, report, test_df, y_true, y_pred):
    residuals = y_true - y_pred
    abs_err = np.abs(residuals)
    pct_err = abs_err / y_true * 100
    cv_mae, cv_rmse, cv_r2 = (np.array(report[k]) for k in ("cv_mae", "cv_rmse", "cv_r2"))

    # Learning curve (one step per training size)
    lc = report["learning_curve"]
    for size, tr, va in zip(lc["train_sizes"], lc["train_mse"], lc["val_mse"]):
        trackio.log(
            {"learning_curve/train_mse": tr, "learning_curve/val_mse": va},
            step=int(size),
        )

    # Titled image versions of the CV and learning-curve charts, so a downloaded
    # image still carries its title (trackio's native line charts do not export one).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    folds = np.arange(1, len(cv_rmse) + 1)
    for ax, values, metric in zip(axes, [cv_mae, cv_rmse, cv_r2], ["MAE", "RMSE", "R2"]):
        ax.bar(folds, values)
        ax.axhline(values.mean(), color="k", linestyle="--", label=f"Mean {values.mean():.2f}")
        ax.set(xlabel="Fold", ylabel=metric, title=f"{metric} per Fold")
        if metric == "R2":
            r2_ylim(ax, values)
        ax.legend()
        ax.grid(True)
    fig.suptitle(f"{name}: 5-Fold Cross-Validation Scores", fontsize=16)
    fig.tight_layout()
    cv_image = to_image(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lc["train_sizes"], lc["train_mse"], marker="o", label="Training MSE")
    ax.plot(lc["train_sizes"], lc["val_mse"], marker="o", label="Validation MSE")
    ax.set(
        xlabel="Training Set Size",
        ylabel="Mean Squared Error",
        title=f"{name}: Learning Curve",
    )
    ax.legend()
    ax.grid(True)
    learning_curve_image = to_image(fig)

    trackio.log({"plots/cv_folds": cv_image, "plots/learning_curve": learning_curve_image})

    # Residual analysis
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=12)
    ax.axhline(0, color="r", linestyle="--")
    ax.set(xlabel="Predicted Price", ylabel="Residual", title="Residuals vs Predicted")
    ax = axes[0, 1]
    ax.hist(residuals, bins=40, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--")
    ax.set(
        xlabel="Residual (Actual - Predicted)",
        ylabel="Count",
        title="Residual Distribution",
    )
    ax = axes[1, 0]
    ax.hist(np.clip(pct_err, 0, 100), bins=40, edgecolor="black")
    ax.axvline(
        np.median(pct_err),
        color="r",
        linestyle="--",
        label=f"Median {np.median(pct_err):.1f}%",
    )
    ax.set(
        xlabel="Absolute % Error (clipped at 100)",
        ylabel="Count",
        title="Percentage Error Distribution",
    )
    ax.legend()
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Residual Q-Q Plot")
    for a in axes.flat:
        a.grid(True)
    fig.suptitle(f"{name}: Residual Analysis (Test Set)", fontsize=16)
    fig.tight_layout()

    trackio.log({"plots/residual_analysis": to_image(fig)})

    err_df = test_df.copy()
    err_df["abs_err"] = abs_err
    err_df["actual"] = y_true
    err_df["predicted"] = y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.scatter(err_df["里程（km）"], err_df["abs_err"], alpha=0.5, s=12)
    ax.set(xlabel="Mileage (km)", ylabel="Absolute Error", title="Absolute Error vs Mileage")
    ax.grid(True)
    err_df.groupby("車型")["abs_err"].mean().sort_values().plot.barh(ax=axes[0, 1])
    axes[0, 1].set(xlabel="MAE", title="MAE by Model")
    err_df.groupby("年份（首次登記）")["abs_err"].mean().plot.bar(ax=axes[1, 0])
    axes[1, 0].set(xlabel="First Registration Year", ylabel="MAE", title="MAE by Year")
    top_countries = err_df["國家"].value_counts().head(10).index
    err_df[err_df["國家"].isin(top_countries)].groupby("國家")["abs_err"].mean().sort_values().plot.barh(ax=axes[1, 1])
    axes[1, 1].set(xlabel="MAE", title="MAE by Country (top 10 by count)")
    fig.suptitle(f"{name}: Error Breakdown by Feature (Test Set)", fontsize=16)
    fig.tight_layout()

    trackio.log({"plots/error_breakdown": to_image(fig)})

    # Largest errors table
    worst = err_df.nlargest(20, "abs_err")
    trackio.log(
        {
            "tables/largest_errors": trackio.Table(
                dataframe=worst[
                    [
                        "車型",
                        "國家",
                        "年份（首次登記）",
                        "里程（km）",
                        "actual",
                        "predicted",
                        "abs_err",
                    ]
                ].reset_index(drop=True)
            )
        }
    )


# ==========================================
# Run Each Model as a Trackio Run
# ==========================================

y_true = y_test.to_numpy()
r2_scores = {}  # name -> (cv r2 per fold, test r2)
metrics = {}  # name -> {"CV MAE": ..., "Test MAE": ..., ...}
reports = {}  # name -> report json (per-fold CV scores)

for name in MODELS:
    model = joblib.load(model_path(name))
    with open(report_path(name), encoding="utf-8") as f:
        report = json.load(f)
    reports[name] = report

    x = test_df[report["columns"]]
    y_pred = model.predict(x)
    assert np.isfinite(y_pred).all(), f"Invalid predictions in {name}"

    trackio.init(
        project=PROJECT,
        name=name,
        config={
            "model": type(model).__name__,
            "features": report["columns"],
            "cv_folds": len(report["cv_r2"]),
        },
    )
    r2_scores[name] = log_common(name, report, y_true, y_pred)
    metrics[name] = {
        "CV MAE": np.mean(report["cv_mae"]),
        "CV RMSE": np.mean(report["cv_rmse"]),
        "CV R2": np.mean(report["cv_r2"]),
        "Test MAE": mean_absolute_error(y_true, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Test R2": r2_score(y_true, y_pred),
    }
    if name in OLS_MODELS:
        log_main(name, report, test_df, y_true, y_pred)
    trackio.finish()


# ==========================================
# 4. R2 Comparison (y-axis extends below 0 so negative R2 is visible)
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, (cv_r2, _) in r2_scores.items():
    axes[0].plot(range(1, len(cv_r2) + 1), cv_r2, marker="o", label=name)
axes[0].set(xlabel="Fold", ylabel="R2", title="CV R2 per Fold")

names = list(r2_scores)
test_values = [r2_scores[n][1] for n in names]
bars = axes[1].bar(names, test_values)
axes[1].bar_label(bars, fmt="%.3f")
axes[1].set(ylabel="R2", title="Test R2")

all_values = np.concatenate([r2_scores[n][0] for n in names] + [test_values])
for ax in axes:
    r2_ylim(ax, all_values)
    ax.grid(True)
axes[0].legend()
fig.suptitle("Model R2 Comparison", fontsize=16)
fig.tight_layout()

trackio.init(project=PROJECT, name="model-comparison")
trackio.log({"plots/r2_comparison": to_image(fig)})
trackio.finish()


# ==========================================
# 5. Metrics Table (console; CV = mean over folds)
# ==========================================

# column -> value format
METRICS = {
    "CV MAE": "{:,.0f}",
    "CV RMSE": "{:,.0f}",
    "CV R2": "{:.3f}",
    "Test MAE": "{:,.0f}",
    "Test RMSE": "{:,.0f}",
    "Test R2": "{:.3f}",
}


def print_table(title, header, rows, text_cols):
    """Print a plain-text table; the first `text_cols` columns are left-aligned, the rest right-aligned."""
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    def format_row(cells):
        return " | ".join(c.ljust(w) if i < text_cols else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    print(f"\n{title}")
    print(format_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row))


print_table(
    "Results:",
    ["Model"] + list(METRICS),
    [[name] + [fmt.format(m[metric]) for metric, fmt in METRICS.items()] for name, m in metrics.items()],
    text_cols=1,
)


# ==========================================
# 6. Validation Tables (console; mean ± std over folds)
# ==========================================

# column -> (report key suffix, value format)
CV_METRICS = {
    "MAE": ("mae", "{:,.0f}"),
    "RMSE": ("rmse", "{:,.0f}"),
    "R2": ("r2", "{:.3f}"),
}


def mean_std(values, fmt):
    values = np.asarray(values)
    return f"{fmt.format(values.mean())} ± {fmt.format(values.std(ddof=1))}"


for title, prefix in [
    ("5-Fold CV (KFold, shuffled, seed 42):", "cv"),
    (f"5-Fold GroupKFold (grouped by {settings['validation']['group-column']}):", "group_cv"),
]:
    names = [n for n in reports if f"{prefix}_mae" in reports[n]]
    missing = [n for n in reports if n not in names]
    print_table(
        title,
        ["Model"] + list(CV_METRICS),
        [[n] + [mean_std(reports[n][f"{prefix}_{key}"], fmt) for key, fmt in CV_METRICS.values()] for n in names],
        text_cols=1,
    )
    if missing:
        print(f"(no {prefix} scores for {', '.join(missing)}; rerun models.py with overwrite.analysis: true)")

print(f"\nLogged to trackio project '{PROJECT}'. Launch the dashboard with:")
print("  python dashboard.py")
