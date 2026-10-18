import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

from data import test_df, y_test
from models import MODELS, OUT_DIR, model_path

# ==========================================
# Significance Settings
# ==========================================

# Compared on test-set MAE: positive difference = model B better
model_a = "ols"
model_b = "baseline-strong"
n_bootstrap = 10000
n_permutations = 10000
alpha = 0.05
seed = 42


# ==========================================
# 1. Load Models from output/
# ==========================================


def ensure_models(names):
    """Offer to run models.py if any of the saved models are missing."""
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(MODELS)}")

    missing = [n for n in names if not model_path(n).exists()]
    if not missing:
        return

    print("Missing trained model(s):")
    for n in missing:
        print(f"  {model_path(n)}")

    if not sys.stdin.isatty():
        sys.exit("Run `python models.py` first.")

    answer = input("Run models.py now to train them? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        sys.exit("Aborted. Run `python models.py` first.")

    subprocess.run([sys.executable, "models.py"], check=True)

    still_missing = [n for n in missing if not model_path(n).exists()]
    if still_missing:
        sys.exit(f"models.py finished but these are still missing: {', '.join(still_missing)}")


def predict(name):
    model = joblib.load(model_path(name))
    _, columns = MODELS[name]
    y_pred = np.asarray(model.predict(test_df[columns])).ravel()
    assert np.isfinite(y_pred).all(), f"Invalid predictions in {name}"
    return y_pred


ensure_models([model_a, model_b])

pred_a = predict(model_a)
pred_b = predict(model_b)


# ==========================================
# 2. Calculate MAE and Error Differences
# ==========================================

y_true = np.asarray(y_test).ravel()

errors_a = np.abs(y_true - pred_a)
errors_b = np.abs(y_true - pred_b)

mae_a = mean_absolute_error(y_true, pred_a)
mae_b = mean_absolute_error(y_true, pred_b)

# Positive difference means model B has lower MAE
error_diff = errors_a - errors_b
observed_diff = error_diff.mean()

print("\n========== Test Results ==========")
print(f"Test Samples: {len(y_true)}")
print(f"{model_a} MAE: {mae_a:.2f}")
print(f"{model_b} MAE: {mae_b:.2f}")
print(f"MAE Difference ({model_a} - {model_b}): {observed_diff:.2f}")


# ==========================================
# 3. Paired Bootstrap (95% CI)
# ==========================================

n_samples = len(y_true)

rng_bootstrap = np.random.default_rng(seed)

bootstrap_diffs = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    indices = rng_bootstrap.choice(n_samples, size=n_samples, replace=True)
    bootstrap_diffs[i] = error_diff[indices].mean()

ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])

print("\n========== Paired Bootstrap ==========")
print(f"Bootstrap Iterations: {n_bootstrap}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")


# ==========================================
# 4. Paired Permutation Test (p-value)
# ==========================================

rng_permutation = np.random.default_rng(seed)

permutation_diffs = np.empty(n_permutations)

for i in range(n_permutations):
    # Randomly swap model labels within each pair
    signs = rng_permutation.choice([-1, 1], size=n_samples)
    permutation_diffs[i] = np.mean(error_diff * signs)

# Two-sided permutation p-value
extreme_count = np.sum(np.abs(permutation_diffs) >= abs(observed_diff))

p_value = (extreme_count + 1) / (n_permutations + 1)

print("\n========== Paired Permutation Test ==========")
print(f"Permutation Iterations: {n_permutations}")
print(f"Two-sided p-value: {p_value:.6f}")

if p_value < alpha:
    print(f"Result: Statistically significant (p < {alpha})")
else:
    print(f"Result: Not statistically significant (p >= {alpha})")


# ==========================================
# 5. Overall Interpretation
# ==========================================

print("\n========== Final Interpretation ==========")

if ci_lower > 0:
    print(f"Bootstrap CI: {model_b} has lower MAE than {model_a}.")
elif ci_upper < 0:
    print(f"Bootstrap CI: {model_a} has lower MAE than {model_b}.")
else:
    print("Bootstrap CI: Includes zero.")

if p_value < alpha:
    if observed_diff > 0:
        print(f"Permutation Test: {model_b} significantly outperforms {model_a}.")
    else:
        print(f"Permutation Test: {model_a} significantly outperforms {model_b}.")
else:
    print("Permutation Test: No significant difference detected.")


# ==========================================
# 6. Bootstrap Distribution Plot
# ==========================================

bootstrap_path = OUT_DIR / "bootstrap_mae_difference.png"

plt.figure(figsize=(9, 5))

plt.hist(bootstrap_diffs, bins=50, alpha=0.75, edgecolor="black")

plt.axvline(0, color="red", linestyle="--", linewidth=2, label="No Difference (0)")
plt.axvline(observed_diff, color="green", linewidth=2, label=f"Observed Difference = {observed_diff:.2f}")
plt.axvline(ci_lower, color="orange", linestyle=":", linewidth=2, label=f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
plt.axvline(ci_upper, color="orange", linestyle=":", linewidth=2)

plt.xlabel(f"MAE Difference ({model_a} - {model_b})")
plt.ylabel("Frequency")
plt.title("Paired Bootstrap Distribution of MAE Difference")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(bootstrap_path, dpi=300, bbox_inches="tight")
plt.close()


# ==========================================
# 7. Permutation Distribution Plot
# ==========================================

permutation_path = OUT_DIR / "permutation_test.png"

plt.figure(figsize=(9, 5))

plt.hist(permutation_diffs, bins=50, alpha=0.75, edgecolor="black")

plt.axvline(observed_diff, color="red", linewidth=2, label=f"Observed Difference = {observed_diff:.2f}")
plt.axvline(-observed_diff, color="red", linestyle="--", linewidth=2, label="Two-sided Threshold")

plt.xlabel("MAE Difference Under Null Hypothesis")
plt.ylabel("Frequency")
plt.title(f"Paired Permutation Test (p = {p_value:.6f})")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(permutation_path, dpi=300, bbox_inches="tight")
plt.close()


print("\n========== Files Saved ==========")
print(bootstrap_path)
print(permutation_path)
print("\nStatistical significance analysis completed!")
