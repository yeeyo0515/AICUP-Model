# AICUP-Model（nnU-Net v2 心臟肌肉分割）

本專案使用 nnU-Net v2 建置 3D CT 影像分割流程，涵蓋資料前處理（planning + preprocessing）、模型訓練（5-fold cross validation）與推論（sliding window inference）。本 README 說明如何建立虛擬環境、設定路徑、放置資料並重現前處理、訓練與預測步驟。

---

## 1. 系統需求

- 作業系統：Ubuntu 20.04 LTS（建議）
- Python：3.9 以上（建議 3.10）
- GPU：建議 NVIDIA GPU + CUDA（無 GPU 亦可執行，但訓練速度會大幅下降）
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

#### 方式 B：使用本專案內的 nnUNet_v2（若你有改動程式碼／要完全重現本 repo 行為）
在專案根目錄執行：
```bash
pip install -e ./nnUNet_v2
pip install -U simpleitk nibabel
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
   ├─ imagesTs/          # 測試影像（無 label）
   └─ dataset.json       # 資料集描述檔（必要）
```

### 4.1 檔名規則（非常重要）

- 影像（CT 單通道）：
  - `imagesTr/CASE_XXXX_0000.nii.gz`
  - `imagesTs/CASE_XXXX_0000.nii.gz`
- 標註（與病例同名，不加 `_0000`）：
  - `labelsTr/CASE_XXXX.nii.gz`

其中 `CASE_XXXX` 必須一致對應。

---

## 5. 前處理（Planning + Preprocessing）

完成資料放置後，執行規劃與前處理。此步驟會產生 plans、resampling 後資料、正規化參數等，輸出到 `$nnUNet_preprocessed`。

```bash
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity
```
---

## 6. 訓練（Training）

nnU-Net v2 通常以 5-fold 方式訓練。以下以 `3d_fullres` 為例。

### 6.1 訓練單一 fold（以 fold 0 為例）
```bash
nnUNetv2_train 003 3d_fullres 0
```

### 6.2 訓練全部 folds（0~4）
```bash
for f in 0 1 2 3 4; do
  nnUNetv2_train 003 3d_fullres $f
done
```

### 6.3 指定 GPU（建議）
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0
```

---

## 7. 推論／預測（Prediction）

推論可使用 `-f all` 進行 5-fold ensemble（需已完成 0~4 folds 訓練）。

### 7.1 以 ensemble 方式推論
```bash
nnUNetv2_predict \
  -i /path/to/input_imagesTs \
  -o /path/to/output_predictions \
  -d 003 \
  -c 3d_fullres \
  -f all
```

### 7.2 指定單一 fold 推論（例如 fold 0）
```bash
nnUNetv2_predict \
  -i /path/to/input_imagesTs \
  -o /path/to/output_predictions \
  -d 003 \
  -c 3d_fullres \
  -f 0
```

---

## 8. 輸入與輸出說明

- 推論輸入資料夾（`-i`）內檔名需符合：
  - `CASE_XXXX_0000.nii.gz`
- 推論輸出資料夾（`-o`）會產生每個病例的分割結果（通常為 `CASE_XXXX.nii.gz`）

---

## 11. 參考與外部資源

- nnU-Net 官方 GitHub：
  https://github.com/MIC-DKFZ/nnUNet
