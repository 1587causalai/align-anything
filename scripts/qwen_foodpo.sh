#!/usr/bin/env bash
#
# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 使用本地Qwen1.5小模型
MODEL_NAME_OR_PATH="/root/models/Qwen1.5-0.5B" # 本地模型路径

# 设置随机端口避免冲突
export MASTER_PORT=29500

# 设置PYTHONPATH和关键路径
SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname "${SCRIPT_DIR}")
export PYTHONPATH="${ROOT_DIR}"

# 准备FOODPO数据集
FOODPO_DATA_DIR="${ROOT_DIR}/data/foodpo_data"
echo "正在准备FOODPO数据集..."
python ${SCRIPT_DIR}/prepare_foodpo_dataset.py --output_dir ${FOODPO_DATA_DIR} --num_examples 10

# 设置数据集路径和输出目录
TRAIN_DATA_FILES="${FOODPO_DATA_DIR}/train.json"
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen_foodpo" # 输出目录

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 设置为离线模式，避免wandb连接问题
export WANDB_MODE=offline

# 创建FOODPO模板，如果目录不存在则创建
TEMPLATES_DIR="${ROOT_DIR}/align_anything/templates"
mkdir -p ${TEMPLATES_DIR}

# 检查FOODPO模板是否存在
if [ ! -f "${TEMPLATES_DIR}/foodpo.py" ]; then
    echo "错误：FOODPO模板不存在，请确保文件路径正确"
    exit 1
fi

# 执行前检查FOODPO训练模块
echo "检查训练模块是否存在..."
if [ ! -f "${ROOT_DIR}/align_anything/trainers/text_to_text/foodpo.py" ]; then
    echo "错误：训练模块不存在，请确保文件路径正确"
    exit 1
fi

echo "开始FOODPO训练..."

# 运行简化版本的训练命令
cd ${ROOT_DIR}
python -m align_anything.trainers.text_to_text.foodpo \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_data_files ${TRAIN_DATA_FILES} \
     --train_template "align_anything.templates.foodpo.FOODPOTemplate" \
     --output_dir ${OUTPUT_DIR} \
     --epochs 1 \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --learning_rate 5e-5 \
     --save_interval 50 