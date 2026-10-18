from pathlib import Path
import pandas as pd
from utils import read_settings

settings = read_settings()


# ==========================================
# Read input
# ==========================================


inp = settings["input"]

numerical_features = inp["features"]["numerical"]
categorical_features = inp["features"]["categorical"]
features = numerical_features + categorical_features

train_df = pd.read_excel(inp["file"], sheet_name=inp["train-sheet"])
test_df = pd.read_excel(inp["file"], sheet_name=inp["test-sheet"])

x_train = train_df[features].copy()
x_test = test_df[features].copy()

y_train = train_df[inp["target"]].copy()
y_test = test_df[inp["target"]].copy()


# ==========================================
# Write to file
# ==========================================

out = settings["output"]
out_dir = Path(out["dir"])
out_dir.mkdir(parents=True, exist_ok=True)

for key, df in [
    ("train-x", x_train),
    ("train-y", y_train),
    ("test-x", x_test),
    ("test-y", y_test),
]:
    path = out_dir / f"{out[key]}.csv"
    if settings["overwrite"]["data"] or not path.exists():
        df.to_csv(path, index=False, encoding="utf-8-sig")


# ==========================================
# Print result
# ==========================================

print("Selected Features:")
print(x_train.columns.tolist())

print("\nTarget:")
print(inp["target"])

print("\nTraining Set:")
print("X_train:", x_train.shape)
print("y_train:", y_train.shape)

print("\nTest Set:")
print("X_test:", x_test.shape)
print("y_test:", y_test.shape)

print("\nFirst 5 rows:")
print(x_train.head())
