# GRPO 简介

## 1. GRPO是什么？

GRPO是一种强化学习算法，主要用于优化策略模型（比如语言模型），以生成高质量的输出（比如文本）。它通过一个**奖励函数**来衡量输出的质量，同时通过一些约束来保证生成内容的合理性和多样性。简单来说，GRPO的目标是：

- **最大化奖励**：让模型生成的文本获得更高的分数。
- **保持稳定性**：避免模型偏离原始行为太多，防止生成不自然或单一的内容。

在代码中，这对应于`GRPOTrainer`类，通过训练`actor_model`来实现这个目标。

---

## 2. 基本概念

在深入数学之前，先理解几个核心概念：

- **策略（Policy）**  
  策略是一个概率分布，决定模型在给定输入（比如提示词）下生成下一个词的概率。在代码中，`actor_model`是策略模型，`generate_completions`方法根据提示生成多个序列。

- **奖励（Reward）**  
  奖励是衡量生成文本质量的指标，由一个单独的`reward_model`计算。比如，奖励可能是文本的流畅度或相关性评分。代码中的`compute_rewards`方法实现了这一步。

- **KL散度（KL Divergence）**  
  KL散度衡量两个概率分布的差异。在GRPO中，它用来约束当前策略（训练中的模型）与参考策略（初始模型`actor_reference_model`）的偏离，避免变化过大。

- **优势函数（Advantage）**  
  优势函数衡量某个生成序列比平均水平好多少或差多少。它帮助模型知道哪些序列值得鼓励，哪些需要减少概率。

---

## 3. GRPO的目标

GRPO的目标可以用一句话概括：  
**在最大化奖励的同时，通过KL散度限制策略偏离，确保生成内容的质量和多样性。**

数学上，这通过一个**损失函数**来实现，代码中的`train_step`方法计算并优化这个损失。

---

## 4. 数学原理：损失函数

GRPO的核心是定义一个损失函数，然后用梯度下降优化它。损失函数的具体形式如下：

\[ \text{loss} = - \left( \exp(\log p - \log p_{\text{detached}}) \cdot \text{advantage} - \beta \cdot \text{KL} \right) \]

这个公式看起来复杂，但我们可以一步步拆解：

### 4.1 组成部分

- **\(\log p\)**  
  当前策略（`actor_model`）对生成序列的对数概率。代码中通过`_get_per_token_logps`计算。

- **\(\log p_{\text{detached}}\)**  
  \(\log p\)的“脱离梯度”版本，值相同但不参与梯度计算，用来稳定训练。

- **\(\text{advantage}\)**  
  优势函数，表示当前序列相对于平均水平的优劣。代码中通过奖励的组内标准化计算：
  \[ \text{advantage} = \frac{\text{reward} - \text{组内均值}}{\text{组内标准差}} \]

- **\(\beta\)**  
  一个超参数，控制KL散度的权重。在代码中是`self.beta`，比如0.1，表示惩罚力度。

- **\(\text{KL}\)**  
  当前策略和参考策略（`actor_reference_model`）之间的KL散度，计算公式是：
  \[ \text{KL} = \exp(\log p_{\text{ref}} - \log p) - (\log p_{\text{ref}} - \log p) - 1 \]
  其中\(\log p_{\text{ref}}\)是参考模型的对数概率。

### 4.2 损失函数的意义

损失函数分为两部分：

#### 第一部分：\(\exp(\log p - \log p_{\text{detached}}) \cdot \text{advantage}\)
- **作用**：鼓励模型增加高质量序列的概率。
- **解释**：
  - \(\exp(\log p - \log p_{\text{detached}})\)实际上是\(\frac{p}{p_{\text{detached}}}\)，但因为\(\log p_{\text{detached}}\)不参与梯度计算，这部分在数值上等于1，但在梯度计算时会推动\(\log p\)朝高优势方向移动。
  - 如果\(\text{advantage} > 0\)（奖励高于平均），损失变负，模型会增加这个序列的概率。
  - 如果\(\text{advantage} < 0\)（奖励低于平均），损失变正，模型会减少这个序列的概率。

#### 第二部分：\(\beta \cdot \text{KL}\)
- **作用**：惩罚策略偏离参考策略太远。
- **解释**：
  - KL散度越大，说明当前策略和初始策略差异越大，惩罚越大。
  - \(\beta\)控制惩罚的强度，确保模型不会因为过度追求奖励而生成奇怪的内容。

### 4.3 为什么加负号？
损失函数前有个负号，因为我们用梯度下降优化，而目标是**最大化奖励**和**最小化KL散度**。负号将问题转化为最小化损失。

---

## 5. 优势函数的计算

代码中，优势函数是这样计算的（`train_step`方法）：

1. 对于每个提示，生成`num_generations`个序列（比如G=4）。
2. 用`reward_model`计算每个序列的奖励。
3. 对这组序列的奖励计算均值和标准差：
   \[ \text{组内均值} = \frac{1}{G} \sum \text{reward}, \quad \text{组内标准差} = \sqrt{\frac{1}{G} \sum (\text{reward} - \text{均值})^2} \]
4. 标准化得到优势：
   \[ \text{advantage} = \frac{\text{reward} - \text{组内均值}}{\text{组内标准差} + 1e-4} \]
   （加1e-4避免除以0）。

这样，每个序列的优势反映了它在同组中的相对质量。

---

## 6. 训练过程

结合代码的`train_step`，GRPO的训练步骤如下：

1. **生成序列**  
   用`actor_model`为每个提示生成G个序列（`generate_completions`）。

2. **计算奖励**  
   用`reward_model`为这些序列打分（`compute_rewards`）。

3. **计算优势**  
   对奖励标准化，得到每个序列的优势。

4. **计算概率**  
   - 用`actor_model`计算生成序列的对数概率\(\log p\)。
   - 用`actor_reference_model`计算参考概率\(\log p_{\text{ref}}\)。

5. **计算损失**  
   对于每个token，计算：
   $$\text{per token loss} = -\left( \exp(\log p - \log p_{\text{detached}}) \cdot \text{advantage} - \beta \cdot \text{KL} \right)$$
   然后用掩码（`completion_mask`）排除无效token（如EOS后的填充），求平均损失。

6. **更新模型**  
   通过反向传播（`actor_model.backward`）和优化（`actor_model.step`）更新`actor_model`。

---

## 7. 总结

GRPO通过一个精心设计的损失函数，平衡了以下两点：
- **奖励最大化**：通过优势函数引导模型生成高质量内容。
- **策略稳定性**：通过KL散度限制模型偏离，确保输出的多样性和合理性。

在代码实现中，这体现在`train_step`的每一步，从生成序列到计算损失，再到更新模型参数。GRPO特别适合需要生成高质量文本的任务，比如对话系统或文本摘要。
