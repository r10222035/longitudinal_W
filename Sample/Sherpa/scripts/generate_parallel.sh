#!/bin/bash
# generate_parallel.sh
# 目的：泛用型 Sherpa 多核生成腳本。
# 執行方式：請在您預期的工作目錄 (Run Directory) 內執行此腳本。
# 語法：/path/to/generate_parallel.sh <YAML_CONFIG_PATH> [EVENTS_PER_JOB] [NUM_JOBS]

# --- 1. 參數解析與驗證 ---
if [ -z "$1" ]; then
    echo "[Error] 必須提供 YAML 設定檔的路徑！"
    echo "Usage: $0 <path_to_yaml> [events_per_job=10000] [num_jobs=8]"
    exit 1
fi

# 將 YAML 路徑轉為絕對路徑，防止 Sherpa -p 切換目錄時找不到檔案
CONFIG_FILE=$(realpath "$1")
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[Error] 找不到設定檔: $CONFIG_FILE"
    exit 1
fi

# 讀取可選參數，若未提供則使用預設值
EVENTS_PER_JOB=${2:-100}
NUM_JOBS=${3:-100}

# --- 2. 建立工作環境 ---
# 自動抓取當前執行指令的所在目錄作為 Sherpa 的 Working Directory
SHERPA_EXEC="/home/r10222035/Software/sherpa-mpi/bin/Sherpa"
RUN_DIR=$PWD
OUTPUT_DIR="./hepmc_data"

# 建立存放 hepmc 的子目錄以保持環境整潔
mkdir -p "$OUTPUT_DIR"

echo "==================================================="
echo "Starting Parallel Event Generation"
echo "Working Directory : $RUN_DIR"
echo "Configuration     : $CONFIG_FILE"
echo "Events per Job    : $EVENTS_PER_JOB"
echo "Number of Cores   : $NUM_JOBS"
echo "Total Events      : $((EVENTS_PER_JOB * NUM_JOBS))"
echo "Output Directory  : $OUTPUT_DIR"
echo "==================================================="

# --- 3. 平行派發任務 ---
for (( i=1; i<=NUM_JOBS; i++ ))
do
    SEED=$i
    HEPMC_FILE="${OUTPUT_DIR}/sample_seed${SEED}.hepmc"
    LOG_FILE="${OUTPUT_DIR}/log_seed${SEED}.txt"
    
    # 執行 Sherpa:
    # -p 指定為當前目錄 (讀取此目錄下的 Process/ 與 Results.db)
    # -f 讀取絕對路徑的 YAML 檔
    $SHERPA_EXEC -p "$RUN_DIR" \
           -f "$CONFIG_FILE" \
           EVENTS=$EVENTS_PER_JOB \
           RANDOM_SEED=$SEED \
           EVENT_OUTPUT="HepMC3_GenEvent[$HEPMC_FILE]" > "$LOG_FILE" 2>&1 &
           
    echo "  -> Submitted Job $i (Seed: $SEED)"
done

echo "All jobs submitted. Waiting for completion..."
wait
echo "Generation completed successfully! Files are in $OUTPUT_DIR"