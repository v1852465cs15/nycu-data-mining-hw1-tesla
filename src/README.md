# Tesla 二手車售價預測

以 `Tesla.xlsx` 的二手車資料訓練線性迴歸（OLS）模型預測售價，並與三個基準模型（baseline）比較，結果記錄到 [trackio](https://github.com/gradio-app/trackio) 儀表板。

所有可調整的項目（資料、特徵、模型變體、輸出位置）都集中在 `settings.yml`，一般使用不需要修改程式碼。

## 專案結構

| 檔案 | 用途 |
|---|---|
| `settings.yml` | 所有設定 |
| `data.py` | 讀取 Excel、選取特徵，將訓練／測試資料存成 CSV |
| `models.py` | 訓練所有模型（OLS 變體 + 基準模型）並存到 `output/` |
| `analyze.py` | 在測試集上評估模型、繪製圖表並記錄到 trackio |
| `dashboard.py` | 開啟 trackio 儀表板 |
| `clear.py` | 刪除本機**所有** trackio 專案 |
| `output/` | 產生的資料、模型與報告 |

## 環境安裝

需要 Python 3.10 以上。

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install pandas openpyxl numpy scipy scikit-learn matplotlib joblib pillow trackio pyyaml
```

## 使用流程

> 所有指令都要在專案資料夾（`settings.yml` 所在處）執行，因為程式以相對路徑讀取設定、字型與資料。

### 0. 準備資料（一般不用執行）

```bash
python data.py
```

讀取 `Tesla.xlsx`，依 `input.features` 選取特徵，將切好的原始資料存到 `output/`：

- `trainX.csv`、`trainY.csv`：訓練集的特徵與售價
- `testX.csv`、`testY.csv`：測試集的特徵與售價

並印出所選特徵、資料筆數與前 5 筆資料，方便確認設定是否正確。

> **一般情況下不需要單獨執行這一步。** `models.py` 與 `analyze.py` 都會自動呼叫 `data.py` 讀取資料（並依 `overwrite.data` 產生 CSV）。只有在想單獨檢查資料或匯出 CSV 時才需要執行。

### 1. 訓練模型

```bash
python models.py
```

會先透過 `data.py` 讀取資料，接著依序訓練：

1. `settings.yml` 中 `models:` 列出的每個 OLS 變體
2. 三個基準模型：
   - `baseline-trivial`：永遠預測訓練集平均售價
   - `baseline-simple`：只用 `baseline.simple-features` 中的特徵做 OLS
   - `baseline-strong`：用全部特徵的隨機森林

每個模型都會以完整訓練集訓練，並另外做 5-fold 交叉驗證（KFold 與依 `validation.group-column` 分組的 GroupKFold），輸出：

- `output/<模型名稱>.joblib`：訓練好的模型（含前處理，可直接輸入原始資料預測）
- `output/<模型名稱>.json`：交叉驗證分數（KFold 與 GroupKFold；OLS 變體另含學習曲線）

### 2. 分析並記錄到 trackio

```bash
python analyze.py
```

載入 `output/` 中的模型，在測試集上計算 MAE、RMSE、R²，並為每個模型建立一個 trackio run：

- **所有模型**：摘要指標、各 fold 分數、實際 vs 預測圖、殘差分布
- **OLS 變體另外有**：學習曲線、交叉驗證圖、殘差分析（含 Q-Q 圖）、依特徵分組的誤差（里程、車型、年份、國家）、誤差最大的 20 筆資料
- **`model-comparison`**：所有模型的 R² 比較圖

> 必須先執行 `models.py`，否則 `analyze.py` 找不到模型檔。

### 3. 開啟儀表板

```bash
python dashboard.py
```

瀏覽器會開啟 `http://127.0.0.1:7860`。圖表圖片在 **Media & Tables** 分頁；R² 比較圖在 `model-comparison` run 底下。

> 請用 `python dashboard.py`，不要用 `trackio show`。部分 Windows 電腦會把 `.js` 檔標成純文字，導致 `trackio show` 只顯示白畫面；`dashboard.py` 已修正此問題。若仍是白畫面，按 **Ctrl + Shift + R** 強制重新整理以清除瀏覽器快取。

## 設定說明（`settings.yml`）

### `input`：資料來源與特徵

```yaml
input:
  file: "Tesla.xlsx"
  train-sheet: "訓練集(3002筆)"
  test-sheet: "測試集(752筆)"
  features:
    numerical:      # 數值特徵：標準化
      - "年份（首次登記）"
      - ...
    categorical:    # 類別特徵：one-hot 編碼
      - "車型"
      - ...
  target: "售價"
```

欄位名稱必須與 Excel 標題**完全相同**。注意標題中的括號是全形 `（）`，不是半形 `()`，打錯會出現 `KeyError`。

### `models`：OLS 變體

```yaml
models:
  ols:
    drop: []
  ols-no-car-model:
    drop:
      - "車型"
```

每個項目是一個 OLS 模型，使用 `input.features` 中的全部特徵，再去掉 `drop` 列出的欄位。名稱會用於輸出檔名與 trackio run 名稱。

例如想比較「不用國家」的效果，加上：

```yaml
  ols-no-country:
    drop:
      - "國家"
```

`drop` 只能移除 `input.features` 中已有的特徵；若名稱不存在，`models.py` 會直接報錯並列出該名稱。

### `baseline`：基準模型

```yaml
baseline:
  simple-features:
    - "里程（km）"
```

`baseline-simple` 使用的特徵。可以是 Excel 中任何數值欄位，不必在 `input.features` 裡。

### `output`：輸出位置

`dir` 是輸出資料夾；`train-x` 等是四個 CSV 的檔名（不含 `.csv`）。

### `overwrite`：是否重新產生

| 設定 | `true` | `false` |
|---|---|---|
| `data` | 每次覆寫 CSV | 檔案已存在就跳過 |
| `model` | 每次重新訓練 OLS 變體的模型權重（`.joblib`） | 模型檔已存在就跳過 |
| `baseline` | 每次重新訓練基準模型的模型權重（`.joblib`） | 模型檔已存在就跳過 |
| `analysis` | 每次重新計算所有模型的交叉驗證報告（`.json`） | 報告已存在就跳過 |

> 修改特徵或 `drop` 後，若對應的 `overwrite` 為 `false`，記得刪除 `output/` 中的舊模型檔與報告，或暫時把 `model`／`baseline` 與 `analysis` 都改成 `true`，否則會沿用舊結果。只修改驗證方式（例如 `validation.group-column`）時，只需把 `analysis` 改成 `true`。

### `trackio`

```yaml
trackio:
  project: "tesla-ols"
  clear-on-analyze: true
```

- `project`：trackio 專案名稱
- `clear-on-analyze`：為 `true` 時，每次執行 `analyze.py` 前會先刪除此專案的所有舊 run，避免儀表板出現重複的紀錄；設為 `false` 可保留歷史紀錄

## 其他

- **清除所有 trackio 資料**：`python clear.py` 會刪除本機**所有** trackio 專案（不只本專案），請小心使用。
- **中文字型**：圖表使用 `fonts/NotoSansTC-Regular.otf`，在 Windows、macOS、Linux 上都能正確顯示中文，不需另外安裝字型。
