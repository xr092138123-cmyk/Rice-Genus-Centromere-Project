#!/bin/bash
#
# 从FASTA到预测结果的一键脚本
# Usage: ./predict_from_fasta.sh <genome.fasta> <model.pt> [output_dir] [bin_size] [threads]
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 参数
GENOME=$1
MODEL=$2
OUTPUT_DIR=${3:-"predictions_output"}
BIN_SIZE=${4:-10000}
THREADS=${5:-8}
THRESHOLD=${6:-0.3}

# 帮助信息
if [ -z "$GENOME" ] || [ -z "$MODEL" ]; then
    echo "从FASTA到着丝粒预测的完整流程脚本"
    echo ""
    echo "用法: $0 <genome.fasta> <model.pt> [output_dir] [bin_size] [threads] [threshold]"
    echo ""
    echo "参数:"
    echo "  genome.fasta  - 输入基因组FASTA文件 (必需)"
    echo "  model.pt      - 训练好的模型文件 (必需)"
    echo "  output_dir    - 输出目录 (默认: predictions_output)"
    echo "  bin_size      - Bin大小(bp) (默认: 10000)"
    echo "  threads       - 线程数 (默认: 8)"
    echo "  threshold     - 预测阈值 (默认: 0.3)"
    echo ""
    echo "示例:"
    echo "  $0 genome.fasta checkpoints/best_model.pt"
    echo "  $0 genome.fasta model.pt my_results 5000 16 0.4"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$GENOME" ]; then
    echo -e "${RED}错误: 基因组文件不存在: $GENOME${NC}"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL${NC}"
    exit 1
fi

# 检查依赖
command -v jellyfish >/dev/null 2>&1 || {
    echo -e "${RED}错误: 未找到jellyfish命令${NC}"
    echo "请安装: conda install -c bioconda jellyfish"
    exit 1
}

command -v python >/dev/null 2>&1 || {
    echo -e "${RED}错误: 未找到python命令${NC}"
    exit 1
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 打印信息
echo "========================================="
echo "  着丝粒预测流程 - Centromere Prediction"
echo "========================================="
echo -e "基因组文件:   ${GREEN}$GENOME${NC}"
echo -e "模型文件:     ${GREEN}$MODEL${NC}"
echo -e "输出目录:     ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Bin大小:      ${GREEN}$BIN_SIZE bp${NC}"
echo -e "线程数:       ${GREEN}$THREADS${NC}"
echo -e "预测阈值:     ${GREEN}$THRESHOLD${NC}"
echo "========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
KMER_DIR="$OUTPUT_DIR/kmer_analysis"
mkdir -p "$KMER_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 步骤1: k-mer统计
echo ""
echo -e "${YELLOW}[步骤 1/4] 计算k-mer频率...${NC}"
for k in 64 128 256 512; do
    echo "  处理 k=$k..."
    
    # 检查是否已存在
    if [ -f "${KMER_DIR}/${k}mer_counts.txt" ]; then
        echo "    发现已存在的k-mer文件，跳过"
        continue
    fi
    
    # 计数k-mers
    jellyfish count \
        -m $k \
        -s 1G \
        -t $THREADS \
        -C \
        -o "${KMER_DIR}/${k}mer.jf" \
        "$GENOME"
    
    # 导出为文本格式
    jellyfish dump \
        "${KMER_DIR}/${k}mer.jf" \
        > "${KMER_DIR}/${k}mer_counts.txt"
    
    # 删除中间文件节省空间
    rm "${KMER_DIR}/${k}mer.jf"
    
    echo "    k=$k 完成"
done
echo -e "${GREEN}✓ k-mer分析完成${NC}"

# 步骤2: 生成特征
echo ""
echo -e "${YELLOW}[步骤 2/4] 生成特征CSV...${NC}"
python "$PROJECT_DIR/src/preprocessing/generate_features.py" \
    --genome "$GENOME" \
    --kmer-dir "$KMER_DIR" \
    --output "${OUTPUT_DIR}/features.csv" \
    --bin-size $BIN_SIZE

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 特征生成失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 特征CSV创建完成${NC}"

# 步骤3: 模型推理
echo ""
echo -e "${YELLOW}[步骤 3/4] 运行模型推理...${NC}"
python "$PROJECT_DIR/src/training/inference.py" \
    --checkpoint "$MODEL" \
    --input "${OUTPUT_DIR}/features.csv" \
    --output "${OUTPUT_DIR}/predictions" \
    --threshold $THRESHOLD

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 模型推理失败${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 推理完成${NC}"

# 步骤4: 生成BED文件
echo ""
echo -e "${YELLOW}[步骤 4/4] 生成BED文件...${NC}"
python "$PROJECT_DIR/src/postprocessing/predictions_to_bed.py" \
    "${OUTPUT_DIR}/predictions/predictions.json" \
    "${OUTPUT_DIR}/predictions/centromeres.bed" \
    --min-prob 0.5

if [ $? -ne 0 ]; then
    echo -e "${RED}警告: BED文件生成失败${NC}"
fi

# 也生成详细的BED文件
python "$PROJECT_DIR/src/postprocessing/predictions_to_bed.py" \
    "${OUTPUT_DIR}/predictions/predictions.json" \
    "${OUTPUT_DIR}/predictions/centromeres_detailed.bed" \
    --min-prob 0.3 \
    --detailed

echo -e "${GREEN}✓ BED文件创建完成${NC}"

# 计算运行时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# 总结
echo ""
echo "========================================="
echo -e "${GREEN}流程完成！${NC}"
echo "========================================="
echo "运行时间: ${MINUTES}分${SECONDS}秒"
echo ""
echo "输出文件:"
echo "  📊 特征CSV:      ${OUTPUT_DIR}/features.csv"
echo "  📈 预测JSON:     ${OUTPUT_DIR}/predictions/predictions.json"
echo "  📋 汇总CSV:      ${OUTPUT_DIR}/predictions/predictions_summary.csv"
echo "  🧬 BED文件:      ${OUTPUT_DIR}/predictions/centromeres.bed"
echo "  🧬 详细BED:      ${OUTPUT_DIR}/predictions/centromeres_detailed.bed"
echo ""
echo "下一步:"
echo "  1. 查看预测结果: cat ${OUTPUT_DIR}/predictions/predictions_summary.csv"
echo "  2. 在IGV中可视化: 加载 ${OUTPUT_DIR}/predictions/centromeres.bed"
echo "  3. 查看详细信息: less ${OUTPUT_DIR}/predictions/predictions.json"
echo "========================================="

# 显示预测区域摘要
if [ -f "${OUTPUT_DIR}/predictions/predictions_summary.csv" ]; then
    echo ""
    echo "预测结果摘要:"
    head -n 6 "${OUTPUT_DIR}/predictions/predictions_summary.csv" | column -t -s,
fi

