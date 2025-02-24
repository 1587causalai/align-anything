# PPO 算法详解：原理与推导

## 基础知识

我们来梳理清楚 PPO（Proximal Policy Optimization）算法的原理和推导过程。PPO 是一种强化学习中的策略梯度方法，旨在通过限制新策略与旧策略的差异来稳定训练过程，同时提高样本效率。我们将从策略梯度的基本原理开始，逐步推导 PPO 的核心公式，并解释其意义。


### 1. 策略梯度方法的基本原理

PPO 是基于策略梯度（Policy Gradient）的方法，因此我们首先需要理解策略梯度的基本思想。强化学习的目标是找到一个策略 \(\pi(a|s)\)，使得智能体在环境中的累积回报（即期望回报）最大化。策略 \(\pi(a|s)\) 表示在状态 \(s\) 下选择动作 \(a\) 的概率分布，通常用参数 \(\theta\) 表示为 \(\pi_\theta(a|s)\)。

策略梯度的目标是通过梯度上升优化策略参数 \(\theta\)，从而最大化期望回报 \(J(\theta)\)。根据策略梯度定理，策略梯度的表达式为：

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]
\]

其中：
- \(\tau = \{s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T\}\) 是智能体按照策略 \(\pi_\theta\) 在环境中生成的一条轨迹。
- \(A_t\) 是优势函数（Advantage Function），表示在状态 \(s_t\) 下选择动作 \(a_t\) 相对于平均行为的优劣。
- \(\nabla_\theta \log \pi_\theta(a_t|s_t)\) 是策略的对数概率的梯度，决定了更新的方向。

**优势函数的定义：** \(A_t = Q(s_t, a_t) - V(s_t)\)，其中：
- \(Q(s_t, a_t)\) 是动作值函数，表示从状态 \(s_t\) 选择动作 \(a_t\) 并随后遵循策略 \(\pi_\theta\) 的期望累积回报。
- \(V(s_t)\) 是状态值函数，表示从状态 \(s_t\) 开始遵循策略 \(\pi_\theta\) 的期望累积回报。

优势函数 \(A_t\) 的作用是衡量动作 \(a_t\) 比平均水平好多少或差多少。为了减少方差，我们通常使用蒙特卡洛方法或时序差分（TD）方法来估计 \(Q(s_t, a_t)\)。例如，可以用实际回报 \(R_t = \sum_{k=t}^T \gamma^{k-t} r_k\) 近似 \(Q(s_t, a_t)\)，其中 \(\gamma\) 是折扣因子。

然而，直接使用回报 \(R_t\) 估计优势函数会导致高方差。为了解决这个问题，PPO 引入了 **广义优势估计（Generalized Advantage Estimation, GAE）**，以平滑未来奖励的估计。

---

### 2. 广义优势估计（GAE）

GAE 的目标是通过权衡偏差（bias）和方差（variance），提供更稳定的优势函数估计。我们先从基本的 TD 误差开始推导。

#### 2.1 TD 误差和 \(k\)-步优势估计

TD 误差 \(\delta_t\) 定义为：
\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]
其中：
- \(r_t\) 是时刻 \(t\) 的即时奖励。
- \(\gamma V(s_{t+1})\) 是下一状态的折扣值。
- \(V(s_t)\) 是当前状态的估计值。

基于 TD 误差，我们可以定义 \(k\)-步优势估计 \(\hat{A}_t^k\)：
\[
\hat{A}_t^k = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
\]
这等价于：
\[
\hat{A}_t^k = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}
\]

- 当 \(k=1\) 时，\(\hat{A}_t^1 = \delta_t\)，这是单步 TD 估计，偏差高但方差低。
- 当 \(k \to \infty\) 时，\(\hat{A}_t^\infty = R_t - V(s_t)\)，这是蒙特卡洛估计，方差高但无偏。

#### 2.2 GAE 的推导

为了在偏差和方差之间取得平衡，GAE 通过对不同 \(k\)-步优势估计进行指数加权平均来定义优势函数。具体地，GAE 使用参数 \(\lambda \in [0, 1]\) 来控制权衡：
\[
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \left( \hat{A}_t^1 + \lambda \hat{A}_t^2 + \lambda^2 \hat{A}_t^3 + \dots \right)
\]

我们将每个 \(\hat{A}_t^k\) 展开：
\[
\hat{A}_t^1 = \delta_t
\]
\[
\hat{A}_t^2 = \delta_t + \gamma \delta_{t+1}
\]
\[
\hat{A}_t^3 = \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}
\]
代入 GAE 表达式：
\[
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \left[ \delta_t + \lambda (\delta_t + \gamma \delta_{t+1}) + \lambda^2 (\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + \dots \right]
\]

按 \(\delta_t, \delta_{t+1}, \dots\) 重新整理：
\[
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \left[ \delta_t (1 + \lambda + \lambda^2 + \dots) + \gamma \delta_{t+1} (\lambda + \lambda^2 + \lambda^3 + \dots) + \gamma^2 \delta_{t+2} (\lambda^2 + \lambda^3 + \lambda^4 + \dots) + \dots \right]
\]

利用几何级数求和公式 \(\sum_{k=0}^\infty \lambda^k = \frac{1}{1 - \lambda}\)（当 \(\lambda < 1\) 时），可以化简为：
\[
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
\]

- 当 \(\lambda = 0\) 时，\(\hat{A}_t = \delta_t\)，退化为单步 TD 估计。
- 当 \(\lambda = 1\) 时，\(\hat{A}_t = \sum_{l=0}^\infty \gamma^l \delta_{t+l} = R_t - V(s_t)\)，退化为蒙特卡洛估计。

通过选择合适的 \(\lambda\)（通常接近 1），GAE 可以在偏差和方差之间取得平衡，提供稳定的优势估计 \(\hat{A}_t\)。

---

### 3. PPO 的目标函数

PPO 的核心思想是通过限制新策略 \(\pi_\theta\) 与旧策略 \(\pi_{\theta_{\text{old}}}\) 的差异来稳定训练过程，避免策略更新过大导致的性能崩溃。PPO 的目标函数是基于 clipped surrogate objective 的，我们逐步推导其形式。

#### 3.1 策略更新的基本目标

策略梯度的目标是最大化期望回报 \(J(\theta)\)。在每次更新时，我们希望新策略 \(\pi_\theta\) 的性能优于旧策略 \(\pi_{\theta_{\text{old}}}\)。为此，我们定义一个代理目标函数（surrogate objective）：
\[
L(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right]
\]
其中：
- \(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\) 是新旧策略的概率比（policy ratio）。
- \(\hat{A}_t\) 是通过 GAE 估计的优势函数。
- \(\hat{\mathbb{E}}_t\) 表示对采样数据的期望估计。

如果 \(\hat{A}_t > 0\)，则增加 \(\pi_\theta(a_t|s_t)\) 的概率；如果 \(\hat{A}_t < 0\)，则减少其概率。然而，直接优化这个目标可能导致新旧策略差异过大，引发训练不稳定。

#### 3.2 PPO-Clip 的 clipped surrogate objective

为了限制策略更新的幅度，PPO-Clip 引入了一个剪切（clipping）机制。目标函数定义为：
\[
\mathcal{L}_{\text{PPO-clip}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \text{clip} \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]
\]
其中：
- \(\epsilon\) 是一个超参数（通常为 0.1 或 0.2），控制剪切范围。
- \(\text{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\) 将概率比 \(r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\) 限制在 \([1 - \epsilon, 1 + \epsilon]\) 范围内。

**理解剪切机制：**
- 如果 \(r_t > 1 + \epsilon\)，说明新策略对动作 \(a_t\) 的概率增加过多，剪切后限制为 \(1 + \epsilon\)。
- 如果 \(r_t < 1 - \epsilon\)，说明新策略对动作 \(a_t\) 的概率减少过多，剪切后限制为 \(1 - \epsilon\)。
- 取 \(\min\) 操作确保目标函数不会因为过大的 \(r_t\) 而过度优化，从而避免策略崩溃。

通过这种方式，PPO-Clip 保证了新策略不会偏离旧策略太远，增强了训练的稳定性。

#### 3.3 值函数的更新

除了策略更新，PPO 还需要更新值函数 \(V_\phi(s)\) 以提供更准确的优势估计。值函数的损失函数定义为：
\[
\mathcal{L}_{\text{critic}}(\phi) = \hat{\mathbb{E}}_t \left[ \left\| V_\phi(s_t) - \hat{R}_t \right\|^2 \right]
\]
其中：
- \(\hat{R}_t = \sum_{i=t}^\infty \gamma^{i-t} r_i\) 是实际的折扣回报估计。
- \(V_\phi(s_t)\) 是值函数的预测值，参数为 \(\phi\)。

通过最小化这个均方误差（MSE）损失，值函数可以更准确地估计状态 \(s_t\) 的期望回报。

---

### 4. PPO 的训练过程

综合以上推导，PPO 的训练过程可以总结为以下步骤：

1. **初始化：** 初始化策略参数 \(\theta_0\) 和值函数参数 \(\phi_0\)。
2. **收集轨迹：** 使用当前策略 \(\pi_{\theta_n}\)，在环境中执行并收集一组轨迹 \(\mathcal{D}_n = \{\tau_i\}\)。
3. **计算回报和优势：**
   - 计算折扣回报 \(\hat{R}_t = \sum_{i=t}^\infty \gamma^{i-t} r_i\)。
   - 使用 GAE 计算优势估计 \(\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}\)。
4. **更新策略：** 通过最大化 PPO-Clip 目标函数更新策略参数：
   \[
   \theta_{n+1} = \arg \max_\theta \mathcal{L}_{\text{PPO-clip}}(\theta)
   \]
5. **更新值函数：** 通过最小化 critic 损失函数更新值函数参数：
   \[
   \phi_{n+1} = \arg \min_\phi \mathcal{L}_{\text{critic}}(\phi)
   \]
6. **重复：** 重复步骤 2-5，直到策略收敛。

---

### 5. PPO 的优点与实际意义

- **稳定性：** PPO 通过剪切机制限制策略更新的幅度，避免了传统策略梯度方法中可能出现的策略崩溃问题。
- **样本效率：** 通过 mini-batch 更新和 GAE 优势估计，PPO 提高了样本利用效率。
- **易于实现：** PPO 的算法结构相对简单，超参数较少，易于在实践中调整和应用。
- **广泛应用：** PPO 在许多 benchmark 任务（如机器人控制、游戏 AI）中表现出色，特别是在需要稳定性和高性能的场景中。

---

### 总结

PPO 算法通过策略梯度和 GAE 提供了稳定的优势估计，并利用 clipped surrogate objective 限制策略更新的幅度。其核心公式包括：
- 策略梯度：\(\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]\)
- GAE 优势估计：\(\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}\)
- PPO-Clip 目标函数：\(\mathcal{L}_{\text{PPO-clip}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t \hat{A}_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]\)
- 值函数损失：\(\mathcal{L}_{\text{critic}}(\phi) = \hat{\mathbb{E}}_t \left[ \left\| V_\phi(s_t) - \hat{R}_t \right\|^2 \right]\)



## 核心难点


### 代理目标函数

关于强化学习中 PPO（Proximal Policy Optimization）算法的一个核心概念：**为什么代理目标函数（surrogate objective）的梯度近似等于策略梯度**，以及如果这个关系不能证明，是否意味着该方法缺乏理论基础。下面我们将逐步推导并解释这个关系的成立，并说明它如何为 PPO 提供坚实的根基。

---

#### 1. 策略梯度的定义

在强化学习中，策略梯度方法的目标是最大化期望回报 \( J(\theta) \)，其中 \(\theta\) 是策略 \(\pi_\theta\) 的参数。期望回报的梯度可以表示为：

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^\pi(s_t, a_t) \right]
\]

- **\(\pi_\theta(a_t|s_t)\)**：策略，给定状态 \(s_t\) 选择动作 \(a_t\) 的概率。
- **\(\tau\)**：采样轨迹，\(\tau = \{s_0, a_0, r_0, s_1, a_1, r_1, \dots\}\)。
- **\(A^\pi(s_t, a_t)\)**：优势函数，表示在状态 \(s_t\) 下选择动作 \(a_t\) 相对于平均行为的优劣。
- **\(\nabla_\theta \log \pi_\theta(a_t|s_t)\)**：策略的对数概率的梯度，表示策略的变化方向。

在实践中，我们通过采样估计这个梯度：

\[
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \hat{A}_{i,t}
\]

其中 \(\hat{A}_{i,t}\) 是对优势函数的估计。

---

#### 2. 代理目标函数的引入

PPO 算法为了提高训练稳定性，引入了代理目标函数 \( L(\theta) \)。其基本形式为：

\[
L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]
\]

- **\(\pi_{\theta_{\text{old}}}\)**：旧策略，即更新前的策略。
- **\(\pi_\theta\)**：新策略，即当前优化的策略。
- **\(A^{\pi_{\theta_{\text{old}}}}(s_t, a_t)\)**：基于旧策略计算的优势函数。
- **\(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\)**：新旧策略的概率比，衡量策略的变化。

重要的是，期望是对旧策略 \(\pi_{\theta_{\text{old}}}\) 采样的轨迹 \(\tau\) 计算的，因为我们在更新时使用的是先前收集的数据。

代理目标函数的设计目标是：通过优化 \( L(\theta) \)，间接优化 \( J(\theta) \)。为此，我们需要验证 \( L(\theta) \) 的梯度是否与 \( J(\theta) \) 的梯度一致。

---

#### 3. 计算代理目标函数的梯度

我们需要证明 \(\nabla_\theta L(\theta)\) 在某种条件下近似等于 \(\nabla_\theta J(\theta)\)。具体来说，我们关注 \(\theta = \theta_{\text{old}}\) 的情况，即新策略刚开始等于旧策略时。

代理目标函数为：

\[
L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]
\]

对其求梯度：

\[
\nabla_\theta L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \nabla_\theta \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right) \right]
\]

因为期望是对 \(\pi_{\theta_{\text{old}}}\) 的，而 \(\pi_{\theta_{\text{old}}}\) 和 \(A^{\pi_{\theta_{\text{old}}}}\) 不依赖于 \(\theta\)，我们可以将梯度移到内部：

\[
\nabla_\theta L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \nabla_\theta \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \right) \right]
\]

计算概率比的梯度：

\[
\nabla_\theta \left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \right) = \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
\]

代入后：

\[
\nabla_\theta L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]
\]

在 \(\theta = \theta_{\text{old}}\) 处，\(\pi_\theta = \pi_{\theta_{\text{old}}}\)，所以：

\[
\nabla_\theta L(\theta) \bigg|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\nabla_\theta \pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]
\]

注意到：

\[
\frac{\nabla_\theta \pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \nabla_\theta \log \pi_{\theta_{\text{old}}}(a_t|s_t)
\]

因此：

\[
\nabla_\theta L(\theta) \bigg|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \nabla_\theta \log \pi_{\theta_{\text{old}}}(a_t|s_t) A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right]
\]

这正是策略梯度 \(\nabla_\theta J(\theta_{\text{old}})\) 的形式！即：

\[
\nabla_\theta L(\theta) \bigg|_{\theta = \theta_{\text{old}}} = \nabla_\theta J(\theta_{\text{old}})
\]

---

#### 4. 近似关系的意义

上述推导表明，在 \(\theta = \theta_{\text{old}}\) 处，代理目标函数的梯度 \(\nabla_\theta L(\theta)\) 等于策略梯度 \(\nabla_\theta J(\theta)\)。这意味着，当新策略 \(\pi_\theta\) 非常接近旧策略 \(\pi_{\theta_{\text{old}}}\) 时，优化 \( L(\theta) \) 的方向与优化 \( J(\theta) \) 的方向一致。这种局部一致性为 PPO 的有效性提供了理论依据。

然而，当 \(\theta\) 偏离 \(\theta_{\text{old}}\) 时，两者的梯度不再完全相等。但 PPO 通过限制策略更新的幅度（例如使用 clipped surrogate objective），确保新策略不会偏离旧策略太远，从而保持这种近似关系的实用性。

---

#### 5. PPO 的剪切机制与稳定性

PPO 的代理目标函数实际上采用了剪切形式：

\[
L^{\text{clip}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \min \left( r_t(\theta) A^{\pi_{\theta_{\text{old}}}}(s_t, a_t), \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right) \right]
\]

其中 \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \)。当 \( r_t(\theta) \) 在 \([1-\epsilon, 1+\epsilon]\) 范围内时，优化方向与 \( L(\theta) \) 一致；超出范围时，剪切机制限制更新步长，避免策略变化过大导致性能崩溃。

这种设计确保了 PPO 在利用梯度近似优化策略时，既保持了正确的优化方向，又控制了更新幅度，从而兼顾效率与稳定性。

---

#### 6. 这个关系是否为 PPO 提供根基？

用户问道，如果这个近似不能证明，PPO 是否没有任何根基。答案是否定的。上述推导明确证明了在 \(\theta = \theta_{\text{old}}\) 处，\(\nabla_\theta L(\theta) = \nabla_\theta J(\theta)\)。这表明代理目标函数的梯度近似策略梯度是有严格数学依据的，为 PPO 的理论基础奠定了基石。

更重要的是，PPO 通过剪切机制将理论近似转化为实践中的可控优化步骤。即使在 \(\theta \neq \theta_{\text{old}}\) 时梯度不再完全相等，PPO 的设计确保了更新过程的安全性和有效性。因此，PPO 不仅有理论根基，而且在实践中表现出色。

---

#### 7. 总结

代理目标函数的梯度近似等于策略梯度的原因在于：在 \(\theta = \theta_{\text{old}}\) 处，\(\nabla_\theta L(\theta)\) 与 \(\nabla_\theta J(\theta)\) 数学上相等。这种局部一致性表明，优化代理目标函数可以有效地近似优化期望回报。PPO 通过剪切机制进一步限制策略更新幅度，确保训练稳定性和性能。这证明了 PPO 方法有坚实的理论基础，而非无根之木。

