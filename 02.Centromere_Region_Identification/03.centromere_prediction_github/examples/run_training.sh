#!/usr/bin/env bash
# 训练脚本 - 着丝粒预测Transformer模型
# 使用方法: bash run_training.sh [数据目录]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-/home/centromere_area_prediction_v1/embedding/multi_k_summary/20251129_093947}"

echo "=========================================="
echo "着丝粒预测Transformer模型训练"
echo "=========================================="
echo "脚本目录: ${SCRIPT_DIR}"
echo "数据目录: ${DATA_DIR}"
echo ""

# 检查数据目录
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_DIR}"
    exit 1
fi

# 检查CSV文件
csv_count=$(find "${DATA_DIR}" -name "*_multi_k_summary.csv" | wc -l)
echo "找到 ${csv_count} 个CSV文件"

if [[ ${csv_count} -eq 0 ]]; then
    echo "[ERROR] 未找到CSV文件"
    exit 1
fi

# 创建输出目录
mkdir -p "${SCRIPT_DIR}/outputs"
mkdir -p "${SCRIPT_DIR}/checkpoints"
mkdir -p "${SCRIPT_DIR}/logs"

# 运行训练
cd "${SCRIPT_DIR}"

echo ""
echo "开始训练..."
echo ""

python3 train.py --data_dir "${DATA_DIR}"

echo ""
echo "训练完成！"
echo "检查点目录: ${SCRIPT_DIR}/checkpoints"
echo "日志目录: ${SCRIPT_DIR}/logs"





