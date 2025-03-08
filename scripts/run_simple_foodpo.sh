#!/usr/bin/env bash
#
# 运行简单的FOODPO训练脚本
#

# 设置路径
ROOT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATA_DIR="${ROOT_DIR}/data/foodpo_data"
OUTPUT_DIR="${ROOT_DIR}/outputs/simple_foodpo"

# 确保数据目录存在
mkdir -p ${DATA_DIR}

# 准备数据
echo "准备食物偏好数据集..."
python ${ROOT_DIR}/scripts/prepare_foodpo_dataset.py --output_dir ${DATA_DIR} --num_examples 10

# 设置参数
MODEL_PATH="/root/models/Qwen1.5-0.5B"
DATA_FILE="${DATA_DIR}/train.json"

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}

# 运行训练脚本
echo "开始运行FOODPO训练..."
python ${ROOT_DIR}/scripts/simple_foodpo.py \
    --model_name ${MODEL_PATH} \
    --data_file ${DATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate 5e-5 \
    --batch_size 1 \
    --num_epochs 1 \
    --beta 0.1

echo "FOODPO训练完成！" 