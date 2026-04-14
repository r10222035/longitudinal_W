#!/bin/bash
# run_delphes_parallel.sh
# 目的：泛用型多核平行執行 Delphes 探測器模擬 (使用 Conda 環境 vbs_wl)。
# 執行方式：請在您預期的工作目錄 (Run Directory) 內執行此腳本。

# --- 0. 初始化並啟動 Conda 環境 ---
# 嚴謹的作法：手動載入 conda 的 bash hook，讓腳本能夠辨識 conda 指令
eval "$(conda shell.bash hook)"
conda activate vbs_wl

# 檢查是否成功切換環境
if [ $? -ne 0 ]; then
    echo "[Error] 無法啟動 Conda 環境 'vbs_wl'，請檢查 Conda 設定。"
    exit 1
fi

# --- 1. 軟體與卡片路徑 (全域設定) ---
# 改為直接呼叫 Conda 環境中的全域執行檔
# 注意：請確認您的 Sherpa 輸出的 HepMC 版本。若為 HepMC2 格式，請改為 DelphesHepMC2
DELPHES_EXEC="DelphesHepMC3" 
DELPHES_CARD="/home/r10222035/longitudinal_W/Sample/Cards/delphes_card.dat"

# --- 2. 建立工作環境 (動態抓取) ---
RUN_DIR=$PWD
HEPMC_DIR="${RUN_DIR}/hepmc_data"
ROOT_OUT_DIR="${RUN_DIR}/delphes_root"

# 嚴格的防呆檢查：確保輸入目錄與檔案存在
if [ ! -d "$HEPMC_DIR" ]; then
    echo "[Error] 找不到 HepMC 目錄: $HEPMC_DIR"
    echo "請確保您在正確的工作目錄內執行此腳本。"
    exit 1
fi

# 將找到的 .hepmc 檔案存入陣列
shopt -s nullglob
HEPMC_FILES=(${HEPMC_DIR}/*.hepmc)
if [ ${#HEPMC_FILES[@]} -eq 0 ]; then
    echo "[Error] $HEPMC_DIR 內沒有找到任何 .hepmc 檔案！"
    exit 1
fi

# 建立輸出目錄
mkdir -p "$ROOT_OUT_DIR"

echo "==================================================="
echo "Starting Parallel Delphes Simulation (Conda: vbs_wl)"
echo "Working Directory : $RUN_DIR"
echo "Input HepMC Dir   : $HEPMC_DIR"
echo "Output ROOT Dir   : $ROOT_OUT_DIR"
echo "Delphes Card      : $DELPHES_CARD"
echo "Delphes Executable: $(which $DELPHES_EXEC)"
echo "Files to process  : ${#HEPMC_FILES[@]}"
echo "==================================================="

# --- 3. 平行派發任務 ---
for hepmc_file in "${HEPMC_FILES[@]}"; do
    # 提取檔名（不含副檔名與路徑），例如 vbs_seed1
    base_name=$(basename "$hepmc_file" .hepmc)
    output_root="${ROOT_OUT_DIR}/${base_name}.root"
    log_file="${ROOT_OUT_DIR}/${base_name}_delphes.log"
    
    echo "  -> Submitting Delphes job for: $base_name"
    
    # 執行 Delphes，並使用 '&' 推入背景平行處理
    $DELPHES_EXEC $DELPHES_CARD "$output_root" "$hepmc_file" > "$log_file" 2>&1 &
done

echo "All jobs submitted. Waiting for completion..."
wait
echo "All Delphes simulations successfully completed! Files are in $ROOT_OUT_DIR"