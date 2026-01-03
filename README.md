# AICUP-Model（nnU-Net v2 心臟肌肉分割）

本專案使用 nnU-Net v2 建置 3D CT 影像分割流程，涵蓋資料前處理（planning + preprocessing）、模型訓練（5-fold cross validation）與推論（sliding window inference）。
本 README 說明如何建立虛擬環境、設定路徑、放置資料並重現前處理、訓練與預測步驟。

---

## 1. 系統需求

- 作業系統：Ubuntu 20.04 LTS（建議）
- Python：3.9 以上（建議 3.10）
- GPU：建議 NVIDIA GPU（至少 RTX 4070 等級或以上）
  - 若使用預設 nnU-Net v2 計畫（plans），VRAM 建議 ≥ 12GB
  - 若使用本專案「16GB 目標」自訂 plans（`nnUNetResEncUNetPlans_16G`），VRAM 建議 ≥ 16GB（例如 RTX 4070 Ti SUPER / RTX 4080 / RTX 4090）
- 影像格式：NIfTI（`.nii.gz`）

---

## 2. 建立虛擬環境與安裝

以下示範使用 Conda。若使用 venv/pip，概念相同。

### 2.1 建立與啟動環境
```bash
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
```

### 2.2 安裝 PyTorch（依 CUDA 版本調整）
請依你的 CUDA 版本安裝對應 PyTorch：
```text
https://pytorch.org/get-started/locally/
```

### 2.3 安裝 nnU-Net v2（擇一）

#### 方式 A：安裝官方 nnunetv2（一般使用者建議）
```bash
pip install -U nnunetv2
pip install -U simpleitk nibabel
```

#### 方式 B：從 GitHub 下載 nnU-Net 原始碼（可作可不做）
若你需要查看/修改原始碼，或希望與特定 commit 行為一致，可使用下列方式下載：
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
---

## 3. 設定 nnU-Net v2 環境變數（必要）

nnU-Net v2 依賴三個環境變數定位資料與輸出位置：

- `nnUNet_raw`：原始資料（已整理為 nnU-Net 格式）
- `nnUNet_preprocessed`：前處理輸出（resampling/normalization/plans）
- `nnUNet_results`：訓練權重、log、推論模型等結果

### 3.1 建立資料夾
```bash
mkdir -p /path/to/nnunet/nnUNet_raw
mkdir -p /path/to/nnunet/nnUNet_preprocessed
mkdir -p /path/to/nnunet/nnUNet_results
```

### 3.2 export 環境變數（當前終端機有效）
```bash
export nnUNet_raw=/path/to/nnunet/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnunet/nnUNet_preprocessed
export nnUNet_results=/path/to/nnunet/nnUNet_results
```

### 3.3 永久生效（建議加入 ~/.bashrc）
```bash
echo 'export nnUNet_raw=/path/to/nnunet/nnUNet_raw' >> ~/.bashrc
echo 'export nnUNet_preprocessed=/path/to/nnunet/nnUNet_preprocessed' >> ~/.bashrc
echo 'export nnUNet_results=/path/to/nnunet/nnUNet_results' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 資料放置格式（nnU-Net v2 Dataset 結構）

假設本任務資料集名稱為 `Dataset003_Myocardium`（ID=003），資料夾需放在 `$nnUNet_raw` 下：

```text
$nnUNet_raw/
└─ Dataset003_Myocardium/
   ├─ imagesTr/          # 訓練影像（CT 通常為單通道）
   ├─ labelsTr/          # 訓練標註（與 imagesTr 同病例名）
   └─ dataset.json       # 資料集描述檔（必要）
```

### 4.1 檔名規則（非常重要）

- 影像（CT 單通道）：
  - `imagesTr/CASE_XXXX_0000.nii.gz`
- 標註（與病例同名，不加 `_0000`）：
  - `labelsTr/CASE_XXXX.nii.gz`

其中 `CASE_XXXX` 必須一致對應。

---

## 5. 前處理（Planning + Preprocessing）

完成資料放置後，執行規劃與前處理。此步驟會產生 plans、resampling 後資料、正規化參數等，輸出到 `$nnUNet_preprocessed`。

> 注意：以下 `-d` 後面的資料集 ID 請以你的 Dataset ID 為準（例如 003 對應 `-d 3`，若你資料集 ID 為 011，則使用 `-d 11`）。

```bash
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
```

---

## 6. 可選：建立「16GB 目標」自訂 Plans（可作可不做）

若你希望使用 ResEnc 規劃器，並以 16GB VRAM 為目標建立計畫檔（例如 `nnUNetResEncUNetPlans_16G`），可先執行：

```bash
nnUNetv2_plan_experiment \
  -d 3 \
  -pl nnUNetPlannerResEncM \
  -gpu_memory_target 16 \
  -overwrite_plans_name nnUNetResEncUNetPlans_16G
```

此步驟完成後，訓練時可透過 `-p nnUNetResEncUNetPlans_16G` 指定使用該 plans。

---

## 7. 訓練（Training）

nnU-Net v2 通常以 5-fold 方式訓練。以下以 `3d_fullres` 為例。

### 7.1 基本訓練（預設 plans / trainer）
```bash
nnUNetv2_train 3 3d_fullres 0
```

### 7.2 訓練全部 folds（0~4）
```bash
for f in 0 1 2 3 4; do
  nnUNetv2_train 3 3d_fullres $f
done
```

### 7.3 使用自訂 Plans + 自訂 Trainer（包含自訂 LR；本專案主要設定）
本專案訓練指令如下（請依你的 Dataset ID 與 fold 調整）：

```bash
nnUNetv2_train 11 3d_fullres 0 \
  -p nnUNetResEncUNetPlans_16G \
  -tr nnUNetTrainer_LROffset_v100
```

- `-p nnUNetResEncUNetPlans_16G`：使用「16GB 目標」的自訂 plans（可選）
- `-tr nnUNetTrainer_LROffset_v100`：自訂 trainer（內含自訂學習率排程；例如 warm-up + cosine/offset 等）

> 注意：若你不是使用 Dataset ID = 11，請將 `11` 改為你的資料集 ID（例如 `3`）。

---

## 8. 推論／預測（Prediction）

推論可使用 `-f all` 進行 5-fold ensemble（需已完成 0~4 folds 訓練）。

### 8.1 以 ensemble 方式推論
```bash
nnUNetv2_predict \
  -i /path/to/input_imagesTs \
  -o /path/to/output_predictions \
  -d 3 \
  -c 3d_fullres \
  -f all
```

### 8.2 指定單一 fold 推論（例如 fold 0）
```bash
nnUNetv2_predict \
  -i /path/to/input_imagesTs \
  -o /path/to/output_predictions \
  -d 3 \
  -c 3d_fullres \
  -f 0
```

---

## 9. 輸入與輸出說明

- 推論輸入資料夾（`-i`）內檔名需符合：
  - `CASE_XXXX_0000.nii.gz`
- 推論輸出資料夾（`-o`）會產生每個病例的分割結果（通常為 `CASE_XXXX.nii.gz`）

---

## 10. 常見問題排除

1. 找不到資料集 / dataset id：
   - 確認 `$nnUNet_raw/DatasetXXX_*/` 存在
   - 確認 `dataset.json` 位於資料夾根目錄
   - 確認 `imagesTr` 與 `labelsTr` 檔名能一一對應

2. 前處理或訓練時 RAM/CPU 使用過高：
   - 減少同時執行程序數
   - 視情況調整 nnU-Net v2 的執行緒參數（例如 preprocess 的 worker 數量）

3. CUDA out of memory：
   - 降低 GPU 上其他程序佔用
   - 不使用 16GB 目標 plans（改用預設 plans）
   - 確認 plans/configuration 與 GPU VRAM 相符

---

## 11. 參考與外部資源

- nnU-Net 官方 GitHub：
  https://github.com/MIC-DKFZ/nnUNet
