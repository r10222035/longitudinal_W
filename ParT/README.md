# Particle Transformer (ParT) 訓練與組態說明

本目錄包含 Particle Transformer 於縱向 $W$ 玻色子極化 (Longitudinal $W$ Polarization) 與背景事件區分任務中的訓練 Configs、數據處理模組與實驗產出路徑說明。

---

## 📌 1. 輸入特徵分類

### 🔹 1.1 高階與低階特徵
ParT 輸入特徵分為兩大類：
- **High-level**：以重建之頂層事件變數（如 $l_1, l_2, j_1, j_2, \text{MET}$ 的 $p_T, \eta, \Delta\phi$）構成固定維度 (5, 6) 序列。
- **Low-level**：以事件中重建出的粒子/成分序列作為輸入。根據粒子來源狀態的編碼細緻度與額外標籤的加入，分化出不同的衍生情況。

---

### 🔹 1.2 Low-level 輸入特徵的衍生情況
Low-level 特徵進一步包含以下三個維度的衍生組合：

#### (1) Low-level Baseline (2 類來源標籤)
將輸入粒子劃分為 **Track** 與 **Calorimeter Tower** 兩類。

#### (2) Refined Low-level (4 類 / 5 類 / 9 類標籤)
將 Delphes 重建之粒子依據其**來源狀態與類別屬性** 進一步精細化編碼：
- **Type (4 維標籤)**：
  細分為 `[0: Track, 1: Tower, 2: Electron, 3: Muon]`。
- **Tag (5 維標籤)**：
  細分為 `[0: Lepton 1, 1: Lepton 2, 2: Jet 1 constituent, 3: Jet 2 constituent, 4: Unassociated]`。
- **Tag + Type 雙重組合 (9 維標籤)**：
  結合 4 維 Type (`Track, Tower, Electron, Muon`) 與 5 維 Tag (`Lepton 1, Lepton 2, Jet 1, Jet 2, Unassociated`) 兩組獨立編碼向量 ($4 + 5 = 9$ 維)。

#### (3) MET 虛擬粒子標籤
在 Low-level 序列中處理 MET 的衍生選項：
- **無 MET (`use_met: False`)**：序列與 One-Hot 標籤僅包含真正重建出的物理粒子成分。
- **有 MET (`use_met: True`)**：於粒子序列末端併入 MET 的運動學數值，並在 One-Hot 標籤向量中加入專屬的 MET 狀態標註位元。

---

### 🔹 1.3 為何 Refined 9ch + MET 的標籤維度是 11 維？
* **9ch 的組成**：由 **Type 標籤組 (4 維)** (`[Track, Tower, Electron, Muon]`) 與 **Tag 標籤組 (5 維)** (`[Lepton 1, Lepton 2, Jet 1, Jet 2, Unassociated]`) 雙重編碼拼接而成 ($4 + 5 = 9$ 維)。
* **開啟 `use_met: True` 時**：為了讓 MET 虛擬粒子在 Type 組與 Tag 組皆能被單獨識別，數據處理器：
  * 在 **Type 標籤組** 中加入 1 個 MET 位元 $\rightarrow$ $4 + 1 = 5$ 維 (`[Track, Tower, Electron, Muon, MET]`)
  * 在 **Tag 標籤組** 中也加入 1 個 MET 位元 $\rightarrow$ $5 + 1 = 6$ 維 (`[Lepton 1, Lepton 2, Jet 1, Jet 2, Unassociated, MET]`)
  * 兩組向量拼接後的總標籤維度即為 **$5 + 6 = 11$ 維**。

---

## 📁 2. 組態檔案命名

`ParT/configs/` 目錄下所有 `.yaml` 檔案均遵循命名格式：
```bash
[task]_[feature_level]_[channel_spec].yaml
```

### 任務 [task]
- `ew_vs_bg`：Electroweak $W^\pm W^\pm jj$ vs. Background (Sherpa / MadGraph 混合背景)
- `ll_lt_vs_tt`：$W^\pm_L W^\pm_L + W^\pm_L W^\pm_T$ vs. $W^\pm_T W^\pm_T$ (簡稱 LX vs TT)
- `ll_vs_lt_tt`：$W^\pm_L W^\pm_L$ vs. $W^\pm_L W^\pm_T + W^\pm_T W^\pm_T$ (簡稱 LL vs TX)

### 特徵層級與來源標籤識別碼 (Feature Specs)
- `high_level`：事件層級高階運動學特徵 (3 維標籤)
- `low_level`：雙通道低階粒子基線 (2 維標籤: Track & Tower)
- `refined_4ch`：4 類 Type 偵測器物件標籤 (Track, Tower, Electron, Muon)
- `refined_4ch_met`：4 類 Type 標籤 + MET 虛擬粒子 (5 維標籤)
- `refined_5ch`：5 類 Tag 物理物件歸屬標籤 (Lepton 1, Lepton 2, Jet 1, Jet 2, Unassociated)
- `refined_5ch_met`：5 類 Tag 標籤 + MET 虛擬粒子 (6 維標籤)
- `refined_9ch`：9 類 Tag+Type 組合標籤
- `refined_9ch_met`：9 類 Tag+Type 組合標籤 + MET 虛擬粒子 (11 維標籤)

---

## 📋 3. Config 檔案

| 任務 (Task) | 特徵層級 (Feature Spec) | 配置文件名 (Config File) | 輸入來源標籤維度 (One-Hot) | MET 虛擬粒子 (use_met) |
|---|---|---|---|---|
| **EW vs BG** | High-level | `ew_vs_bg_high_level.yaml` | 3 | False |
| | Low-level Baseline | `ew_vs_bg_low_level.yaml` | 2 | False |
| | Refined 4ch (Type) | `ew_vs_bg_refined_4ch.yaml` | 4 | False |
| | Refined 4ch + MET | `ew_vs_bg_refined_4ch_met.yaml` | 5 | True |
| | Refined 5ch (Tag) | `ew_vs_bg_refined_5ch.yaml` | 5 | False |
| | Refined 5ch + MET | `ew_vs_bg_refined_5ch_met.yaml` | 6 | True |
| | Refined 9ch (Tag+Type) | `ew_vs_bg_refined_9ch.yaml` | 9 | False |
| | Refined 9ch + MET | `ew_vs_bg_refined_9ch_met.yaml` | 11 | True |
| **LL+LT vs TT** | High-level | `ll_lt_vs_tt_high_level.yaml` | 3 | False |
| | Low-level Baseline | `ll_lt_vs_tt_low_level.yaml` | 2 | False |
| | Refined 4ch (Type) | `ll_lt_vs_tt_refined_4ch.yaml` | 4 | False |
| | Refined 4ch + MET | `ll_lt_vs_tt_refined_4ch_met.yaml` | 5 | True |
| | Refined 5ch (Tag) | `ll_lt_vs_tt_refined_5ch.yaml` | 5 | False |
| | Refined 5ch + MET | `ll_lt_vs_tt_refined_5ch_met.yaml` | 6 | True |
| | Refined 9ch (Tag+Type) | `ll_lt_vs_tt_refined_9ch.yaml` | 9 | False |
| | Refined 9ch + MET | `ll_lt_vs_tt_refined_9ch_met.yaml` | 11 | True |
| **LL vs LT+TT** | High-level | `ll_vs_lt_tt_high_level.yaml` | 3 | False |
| | Low-level Baseline | `ll_vs_lt_tt_low_level.yaml` | 2 | False |
| | Refined 4ch (Type) | `ll_vs_lt_tt_refined_4ch.yaml` | 4 | False |
| | Refined 4ch + MET | `ll_vs_lt_tt_refined_4ch_met.yaml` | 5 | True |
| | Refined 5ch (Tag) | `ll_vs_lt_tt_refined_5ch.yaml` | 5 | False |
| | Refined 5ch + MET | `ll_vs_lt_tt_refined_5ch_met.yaml` | 6 | True |
| | Refined 9ch (Tag+Type) | `ll_vs_lt_tt_refined_9ch.yaml` | 9 | False |
| | Refined 9ch + MET | `ll_vs_lt_tt_refined_9ch_met.yaml` | 11 | True |

---

## 🚀 4. 訓練與執行指令 (Execution Commands)

訓練時請確保啟用 Conda 環境 `PyTorch_Env`。

```bash
# 1. 啟用 Conda 環境
conda activate PyTorch_Env

# 2. 執行 ParT 訓練 (以 Refined 4ch + MET 為例)
python ParT/train.py --config ParT/configs/ew_vs_bg_refined_4ch_met.yaml

# 3. 指定特定 GPU 執行
CUDA_VISIBLE_DEVICES=0 python ParT/train.py --config ParT/configs/ll_vs_lt_tt_refined_5ch_met.yaml
```

---

## 📊 5. 訓練結果與輸出目錄結構 (Outputs Structure)

訓練過程與模型產出將自動保存於 `ParT/results/<output_dir>/<weight_strategy>/` 目錄下：

```text
ParT/results/<output_dir>/<weight_strategy>/
├── cv_summary.json                   # 5-Fold 交叉驗證之整體平均 Performance 彙整報告
└── fold_X/                           # 各 Fold 分別之實驗結果 (X 為 0 到 4)
    ├── checkpoints/
    │   └── best_model.pt             # 該 Fold 驗證集 Loss 最低之最佳模型權重檔
    ├── training_history.json         # 包含逐 Epoch 之 loss, val_loss, val_roc_auc, test_roc_auc 紀錄
    ├── loss_history.pdf              # 訓練與驗證 Loss 變化趨勢圖
    ├── auc_history.pdf               # 訓練、驗證與測試 ROC AUC 變化趨勢圖
    └── score_distribution.pdf        # 測試集模型預測得分分佈圖
```
