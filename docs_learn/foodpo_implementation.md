# FOODPO: 食物偏好优化算法实现

## 1. 概述

FOODPO (Food Preference Optimization) 是一个基于DPO (Direct Preference Optimization) 算法的食物偏好优化实现。该算法旨在通过人类偏好数据训练语言模型，使其能够提供更符合健康饮食原则的食物建议。

本文档记录了FOODPO算法的实现过程、关键组件和使用方法。

## 2. 实现架构

FOODPO的实现分为两个版本：

1. **基于Align-Anything框架的完整版本**：复用了DPO的架构，但由于依赖复杂，在快速试错阶段不易使用。
2. **独立简化版本**：一个轻量级实现，不依赖于Align-Anything框架，适合快速试错和概念验证。

### 2.1 简化版FOODPO架构

简化版FOODPO主要包含以下组件：

- **数据准备脚本** (`prepare_foodpo_dataset.py`): 创建食物偏好数据集
- **训练脚本** (`simple_foodpo.py`): 实现DPO算法的核心逻辑
- **运行脚本** (`run_simple_foodpo.sh`): 简化运行流程的shell脚本

## 3. 数据格式

FOODPO使用的数据格式为JSONL（每行一个JSON对象），每个样本包含三个关键字段：

```json
{
  "input": "用户的食物相关问题",
  "chosen": "健康、科学的食物建议",
  "rejected": "不太健康或不够科学的食物建议"
}
```

示例：
```json
{"input": "哪些食物对健康有益？", "chosen": "水果、蔬菜、全谷物、瘦肉和鱼类都是对健康有益的食物。它们富含维生素、矿物质和纤维，有助于维持身体健康。", "rejected": "快餐、甜食和加工食品都挺好的，偶尔吃一些没关系的。"}
```

## 4. 算法核心

FOODPO的核心是DPO算法，其关键步骤包括：

1. **加载模型**：加载预训练语言模型和参考模型（两者初始权重相同）
2. **准备数据**：加载偏好数据集，包含用户问题、优选回答和次选回答
3. **计算损失**：
   - 计算当前模型对优选和次选回答的对数概率
   - 计算参考模型对优选和次选回答的对数概率
   - 计算对数概率比率
   - 应用DPO损失函数：`-log(sigmoid(β * (better_ratio - worse_ratio)))`
4. **优化模型**：通过梯度下降优化当前模型，使其更倾向于生成健康的食物建议

核心损失函数实现：

```python
# 计算log比率
better_log_ratio = better_log_probs.sum() - ref_better_log_probs.sum()
worse_log_ratio = worse_log_probs.sum() - ref_worse_log_probs.sum()

# 应用DPO损失函数
loss = -torch.nn.functional.logsigmoid(beta * (better_log_ratio - worse_log_ratio))
```

## 5. 使用方法

### 5.1 准备环境

确保安装了必要的依赖：

```bash
pip install torch transformers
```

### 5.2 准备数据

使用数据准备脚本创建示例数据：

```bash
python scripts/prepare_foodpo_dataset.py --output_dir data/foodpo_data --num_examples 10
```

### 5.3 运行训练

使用运行脚本一键执行训练：

```bash
./scripts/run_simple_foodpo.sh
```

或者直接运行训练脚本：

```bash
python scripts/simple_foodpo.py \
    --model_name /root/models/Qwen1.5-0.5B \
    --data_file data/foodpo_data/train.json \
    --output_dir outputs/simple_foodpo \
    --learning_rate 5e-5 \
    --batch_size 1 \
    --num_epochs 1 \
    --beta 0.1
```

### 5.4 训练输出

训练完成后，模型会保存在指定的输出目录中：

```
outputs/simple_foodpo/foodpo_model_epoch_1/
```

## 6. 关键参数

- **model_name**: 预训练模型路径或名称
- **data_file**: 训练数据文件路径
- **output_dir**: 输出目录
- **learning_rate**: 学习率，默认5e-5
- **batch_size**: 批次大小，默认1
- **num_epochs**: 训练轮数，默认1
- **beta**: DPO损失的缩放系数，默认0.1

## 7. 注意事项和优化方向

1. **数据质量**：FOODPO的效果很大程度上依赖于偏好数据的质量，建议使用专业营养学知识构建数据集
2. **计算资源**：简化版实现适合小模型，对于大模型训练可能需要更多优化
3. **评估方法**：目前缺乏专门的评估方法，未来可以添加食物建议质量的评估指标
4. **领域扩展**：可以扩展到更多食物相关领域，如烹饪方法、饮食习惯等

## 8. 未来工作

1. **扩充数据集**：收集更多真实的食物偏好数据
2. **模型评估**：开发专门的评估方法来衡量模型提供健康食物建议的能力
3. **多语言支持**：扩展到更多语言
4. **特定领域优化**：针对特定饮食需求（如糖尿病、高血压等）进行优化

## 9. 结论

FOODPO算法通过DPO方法成功实现了食物偏好优化，使模型能够提供更健康、更科学的食物建议。简化版实现提供了一个良好的起点，可以根据需要进一步扩展和优化。 