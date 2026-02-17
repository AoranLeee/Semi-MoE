# 修改方案

## 多任务学习方向

改为共享编码器，增加特征选择模块

设输入数据大小：
$$
X \in \mathbb{R}^{B\times C\times H\times W}
$$
编码器由 L=5 个尺度组成，设共享编码器输出：
$$
\{ f_1, f_2, f_3, f_4, f_5 \} = \mathcal{E}(\mathbf{x})
$$
其中：
$$
f_i \in \mathbb{R}^{B \times C_i \times H_i \times W_i}, \quad H_i = \frac{H}{2^{i-1}},\; W_i = \frac{W}{2^{i-1}}
$$

> 标准 UNet encoder 的多尺度输出。

定义任务集合：
$$
{T} = \{ \text{seg}, \text{sdf}, \text{bnd} \}
$$



### 方案一：尺度内 Task-wise DWConv 特征选择（Scale-wise Task-aware Feature Selection）

**核心思想**：

> 在**每一个尺度内部**，为**每一个任务**学习一个**逐像素权重图**，用于选择该尺度中“哪些位置、哪些语义通道对该任务更重要”。

#### 1 尺度内 Task-wise DWConv

对任意尺度 i和任务 $$t \in \mathcal{T}$$，定义一个 **Depthwise Convolution Selector**：
$$
w_i^{(t)} = \mathcal{DWConv}_i^{(t)}(f_i)
$$
输出权重图：
$$
w_i^{(t)} \in \mathbb{R}^{B \times 1 \times H_i \times W_i}
$$


> **关键点**：
>
> - 权重是 **逐像素（pixel-wise）**
> - 不改变空间分辨率
> - 不混合通道（DWConv）



#### 2 尺度内特征调制（Feature Modulation）

将权重作用到原始特征上：
$$
f_i^{(t)} = w_i^{(t)} \odot f_i
$$
其中：

- $\odot$表示逐元素广播乘法

结果特征维度保持不变：
$$
f_i^{(t)} \in \mathbb{R}^{B \times C_i \times H_i \times W_i}
$$

#### 3 decoder使用

对每个任务 t，最终送入 decoder 的特征集合为：
$$
{\mathcal{F}}^{(t)} = \{ \hat f_1^{(t)}, \hat f_2^{(t)}, \hat f_3^{(t)}, \hat f_4^{(t)}, \hat f_5^{(t)} \}
$$
随后进入任务专属 decoder：
$$
\mathbf{y}^{(t)} = \mathcal{D}^{(t)}\left( \hat{\mathcal{F}}^{(t)} \right)
$$
其中：

- $\mathcal{D}^{(t)}$：UNet Decoder（结构相同，参数不共享）
- $\mathbf{y}^{(t)}$：任务输出（seg mask / sdf / boundary）

#### 对比

原始 UNet（单任务）
$$
\mathbf{y} = \mathcal{D} \big( \{ f_i \}_{i=1}^5 \big)
$$


本方案（多任务）
$$
\mathbf{y}^{(t)} = \mathcal{D}^{(t)} \big( \{ w_i^{(t)} \odot f_i \}_{i=1}^5 \big)
$$
👉 **区别仅在于**：
 在 encoder 与 decoder 之间插入了一个 **task-conditioned multiplicative gating**。

> 我们提出了一种尺度感知、任务感知的特征选择模块，该模块通过轻量级的深度卷积来调整共享编码器特征，使每个任务能够在每个分辨率下有选择地关注空间相关信息。

#### 次生方案

方案一 + f4/f5 约束，f4和f5属于高层语义，特征提取或许意义不大，所以采取不同方式特征提取

f1-f3:用方案一

f4-f5:用方案二

方案二gating设计中，除了选择后的特征图，还可以加上原始特征图，实现专家间全局信息共享

---

### 方案二：Low-rank approximation of task-wise feature selection

记得继续确定K的取值

对 **每一个尺度 $f_i$**，我们做如下操作：

> **共享 K 个尺度专家（DWConv） + 任务条件 gating**

对单尺度（第 i 层）结构设计：

#### 1 共享专家（Shared Experts）

为第 i 个尺度定义 **K 个共享 DWConv 专家**：
$$
e_{i,k} = \phi_{i,k}(f_i), \quad k = 1,\dots,K
$$
其中：

- $\phi_{i,k}$：$Depthwise Conv（kernel = 3×3）$
- 输出维度不变：

$$
e_{i,k} \in \mathbb{R}^{B \times C_i \times H_i \times W_i}
$$



> 🔑 这些专家 **不区分任务**，它们学习的是：

- 边缘型
- 区域型
- 方向性
- 空洞结构
- 连通性
  等**通用医学视觉原语**

具体实际，设K=4，增加 **专家熵 / 利用率的 logging 公式**作为指标：

**方案 2-A：无先验专家（最干净）**

每个尺度 i：
$$
\{ e_{i,1}, e_{i,2}, \dots, e_{i,K} \}, \quad e_{i,k} = \text{DWConv}_{3\times3}(f_i)
$$
特点：

- 完全数据驱动
- 与 Low-Rank Experts 原文最一致
- 消融时最容易说明“**专家组合能力**”

**方案 2-B：弱先验专家（可解释性更强）**

给不同专家不同 kernel / dilation：

| Expert     | 设计                   | 语义        |
| ---------- | ---------------------- | ----------- |
| e₁         | DWConv 3×3             | 局部纹理    |
| e₂         | DWConv 3×3, dilation=2 | 中尺度结构  |
| e₃         | DWConv 5×5 或堆叠      | 平滑 / 区域 |
| e₄（可选） | identity               | 原始特征    |

特点：

- 专家语义可解释
- 对 reviewer 友好
- 但稍微引入 inductive bias

📌 **可作为补充实验**

#### 2 任务 gating（Task-conditioned Mixing）

对于每个任务 $t \in \{\text{seg}, \text{sdf}, \text{bnd}\}$，引入一个轻量 gating：

**gating 输入**

你有三种合理选择（按推荐度）：

**推荐 A（最稳）**：
$$
g_i^{(t)} = \text{Conv}_{1\times1}^{(t)}(f_i)
$$
**可选 B（更轻）**：
$$
g_i^{(t)} = \text{MLP}(\text{GAP}(f_i))
$$
**可选 C（task embedding）**：
$$
g_i^{(t)} = \text{MLP}([ \text{GAP}(f_i), \mathbf{e}_t ])
$$
**gating 输出**
$$
\alpha_i^{(t)} = \text{softmax}_k(g_i^{(t)}) \in \mathbb{R}^{B \times K \times H_i \times W_i}
$$


- softmax 在 **K 维度**,表示：**该任务在该像素处“调用哪些专家”**

#### 3 Low-rank 专家融合（关键公式）

$$
w_i^{(t)} = \sum_{k=1}^{K} \alpha_{i,k}^{(t)} \odot e_{i,k}
$$

得到：
$$
w_i^{(t)} \in \mathbb{R}^{B \times C_i \times H_i \times W_i}
$$
这是 **任务 t 在尺度 i 上的“选后特征”**。

#### 4 跨尺度整合（与方案一完全一致）

你有两种方式（与方案一兼容）：

✔️ 方式 A：直接送入 task decoder
$$
\{w_i^{(t)}\}_{i=1}^{5} \rightarrow \text{UNetDecoder}^{(t)}
$$
每个任务一个 decoder（结构相同，参数不共享）。



✔️ 方式 B：只在 f1–f3 使用 LRE（推荐）
$$
\tilde f_i^{(t)} = \begin{cases} w_i^{(t)}, & i \le 3 \\ f_i, & i > 3 \end{cases}
$$
再送入 decoder，**稳定性更强**。



#### 损失函数必须额外考虑的点

✅ (A) 专家多样性正则（**强烈建议**）

正交 / 相关性约束
$$
{L}_{orth} = \sum_i \sum_{k \neq k'} \frac{ \langle e_{i,k}, e_{i,k'} \rangle }{\|e_{i,k}\|\|e_{i,k'}\|}
$$


- 防止专家趋同
- Low-Rank Experts / MoE 常用

权重建议：
$$
\lambda_{orth} \approx 10^{-3}
$$

------

✅ (B) gating 熵正则（比方案一更重要）
$$
{L}_{ent}^{gate} = \sum \text{Entropy}(\alpha^{(t)})
$$
否则：

- gating 会选同一个专家
- MoE 名存实亡

#### stop-gradient：

方案 2-A（推荐）：

> **gating 不反向更新 encoder**

**好处**：

- encoder 学“通用表征”
- gating 学“任务路由”
- 稳定性显著提高（这是 MoE 实战经验）

---

### 方案三：Patcher-style Decoder（Single-stage Task-wise Gated Fusion）

#### 1 空间对齐

选择一个**统一目标尺度**（推荐）：
$$
H^\* = H_1,\quad W^\* = W_1
$$
对每个尺度：
$$
\tilde f_i = \text{Up}(f_i) \in \mathbb{R}^{B \times C_i \times H^\* \times W^\*}
$$
📌 上采样方式：

- bilinear / nearest
- **无参数，梯度稳定**

通道对齐（可选但推荐）

为了避免后续 gating 参数爆炸：
$$
f_i = \phi_i(\tilde f_i), \quad \phi_i = \text{1×1 Conv},\quad \hat f_i \in \mathbb{R}^{B \times C \times H^\* \times W^\*}
$$

#### 2 尺度拼接（Patcher-style aggregation）

$$
F = \text{Concat}(\hat f_1, \hat f_2, \dots, \hat f_5) \in \mathbb{R}^{B \times (5C) \times H^\* \times W^\*}
$$

这里的 F 就是 **“Patcher 输入 token map” 的等价形式**：

- 每个空间位置 = 一个 multi-scale token
- token 含 5 个尺度的信息

#### 3 Task-wise DWConv Gating（核心）

对每个任务 $t \in \{\text{seg, sdf, bnd}\}$，定义一个 gating：
$$
G^{(t)} = \text{DWConv}^{(t)}(F) \in \mathbb{R}^{B \times (5C) \times H^\* \times W^\*}
$$
特点：

- **Depthwise**：每个通道独立建模
- **逐像素**：空间自适应
- **任务条件化**

**gating 归一化（推荐）**

对尺度维度做 softmax：

将$G^{(t)}$reshape 为：
$$
G^{(t)} \rightarrow \mathbb{R}^{B \times C \times 5 \times H^\* \times W^\*}
$$
然后：
$$
\alpha_{i}^{(t)} = \text{Softmax}_i(G^{(t)})
$$

#### 4 任务特定特征融合

对每个任务：
$$
F^{(t)} = \sum_{i=1}^5 \alpha_i^{(t)} \odot \hat f_i \quad\in \mathbb{R}^{B \times C \times H^\* \times W^\*}
$$
解释：

- 每个像素
- 对 5 个尺度自适应加权
- 权重随任务变化

📌 **这一步完全等价于 Patcher 的 token mixing，但更轻**

#### 5 单阶段解码（极简）

你提出的是：

> 「直接一次 Up_conv + 输出头」

单层 refinement（可选）
$$
\bar F^{(t)} = \text{Conv}_{3\times3}(F^{(t)})
$$
输出头
$$
y^{(t)} = \text{Head}^{(t)}(\bar F^{(t)})
$$


- seg：sigmoid / softmax
- sdf：regression head
- bnd：binary / multi-class

```markdown
x
 ↓
Shared U-Net Encoder
 ↓
{f1,f2,f3,f4,f5}
 ↓ (upsample + 1×1)
{f̂1,...,f̂5}
 ↓ concat
F (multi-scale token map)
 ↓ DWConv gating (task-wise)
{αseg, αsdf, αbnd}
 ↓ weighted sum
{Fseg, Fsdf, Fbnd}
 ↓ single conv + head
{ŷseg, ŷsdf, ŷbnd}
```

#### 损失函数必须额外考虑的点

✅ (A) 任务权重必须动态 / 可学习

你现在用的：

```
MultiTaskLoss (uncertainty-based)
```

👉 **非常合适方案三**

理由：

- 自动抑制不稳定任务
- 防止某任务主导 encoder

✅ (B) gating 正则必须保留

至少要有：

- 熵正则
- 尺度平滑

否则 Patcher gating 会退化为：

> “固定线性投影”

#### stop-gradient

建议：

> **gating 输入对 encoder stop-grad**

理由：

- 否则 encoder 会“迎合 gating”
- 单阶段 decoder 非常容易不稳定

---



### 实验命名

#### **E1 – TaskDW (Full-scale Task-wise Selection)**

**方案一（基线）**

> f1–f5 全尺度 Task-wise DWConv 特征选择

**关键词**

- Task-wise
- Scale-wise
- No expert sharing

**论文写法**

> *E1: Task-wise Depthwise Convolution on All Encoder Scales*

------

#### **E2 – LRE-Free (Low-Rank Experts, No Prior)**

**方案二（无先验专家）**

> 共享专家 + 任务 gating
> encoder stop-grad + expert 正则 + gating 正则

**关键词**

- Low-Rank Experts
- No prior
- Structure-only

**论文写法**

> *E2: Low-Rank Experts without Expert Prior*

------

#### **E3 – LRE-Prior (Low-Rank Experts with Weak Prior)**

**方案二（弱先验专家）**

> 在 E2 基础上加弱 inductive bias

**论文写法**

> *E3: Low-Rank Experts with Weak Task Prior*

------

#### **E4 – Hybrid-High (DW + LRE on High-level Scales)**

**混合方案（f5 使用专家）**

> f1–f4：TaskDW
> f5：Low-Rank Experts

**论文写法**

> *E4: Hybrid Feature Selection with Experts on High-level Scales*

------

#### **E5 – Hybrid-MidHigh (DW + LRE on Mid/High Scales)**

**混合方案（f4–f5 使用专家）**

> f1–f3：TaskDW
> f4–f5：Low-Rank Experts

**论文写法**

> *E5: Hybrid Feature Selection with Experts on Mid-to-High Scales*

------

#### **E6 – PatchGate (Patcher-style Single-stage Decoder)**

**方案三**

> 单阶段 Task-wise gated fusion
> encoder stop-grad

**论文写法**

> *E6: Single-stage Task-wise Gated Decoder*

------

💡 **建议**

- 日志目录：`logs/E1_TaskDW/`
- 模型名：`model_E2_LREFree.pth`
- 表格行名：`E4 (Hybrid-High)`

---

### 代码

#### 🌿 `feat-select-base`

**公共基础分支（非常关键）**

在这里：

- 加 `TaskDWSelector`
- 加 `DWExpert`
- 加 `TaskGating`
- 加 `FeatureAggregator`
- **不接入训练流程**

👉 后面所有实验都从这切

------

#### 🌿 `exp-E1-taskdw`

- 启用 TaskDWSelector
- f1–f5
- 不用专家

------

#### 🌿 `exp-E2-lre-free`

- K experts
- gating
- encoder stop-grad
- 正则

------

#### 🌿 `exp-E3-lre-prior`

- 在 E2 基础上加 prior
- 只动 selector / gating

------

#### 🌿 `exp-E4-hybrid-high`

- f1–f4 TaskDW
- f5 LRE

------

#### 🌿 `exp-E5-hybrid-mid-high`

- f1–f3 TaskDW
- f4–f5 LRE

------

#### 🌿 `exp-E6-patchgate`

- 单阶段 decoder
- 不走 unet decoder

二、必须预留的「配置接口」

config.yaml / args

```
FEATURE_SELECT:
  ENABLE: true
  TYPE: task_dw | lowrank | hybrid | patchgate
  EXPERT_K: 3
  STOP_GRAD_ENCODER: true
  SCALE_MODE: full | high | mid_high
```

------

train.py 中统一入口


**架构工程层面的关键步骤**。

> ✅ 建一个纯“能力分支”
> ❌ 不改原训练流程
> ❌ 不接入 forward 主干
> ✅ 只提供可复用模块

后面 E1–E6 全部从这个分支切。

下面给你一个**工程级设计方案**（接口 + 结构 + forward 逻辑）参考，保证：

* 可扩展到 3 个方案
* 支持 stop-grad
* 支持 low-rank 专家
* 支持 gating 正则
* 不和现有模型耦合

---
# 一、目录结构建议

在 models 下新建：

```text
models/
    modules/
        feat_select/
            __init__.py
            dwconv.py
            task_dw_selector.py
            task_gating.py
            feature_aggregator.py
```

---

# 二、模块 1：DWConv

## 功能

最基础构件：

* depthwise 3×3 conv
* 可选 BN
* 可选 activation
* 保持输入输出 channel 一致

---

## 设计

```python
# dwconv.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    """
    Basic Depthwise Convolution block.

    Args:
        in_channels (int)
        kernel_size (int)
        use_bn (bool)
        activation (str): 'relu' | 'gelu' | None
    """

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        use_bn=True,
        activation='relu'
    ):
        super().__init__()

        padding = kernel_size // 2

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=not use_bn
        )

        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
```

---

# 三、模块 2：TaskDWSelector（方案一核心）

## 功能

> 每个任务一个 DWConv
> 输出作为权重图 or 直接加权

支持：

* task 数量
* stop_grad encoder feature
* 输出权重 or 输出重加权特征

---

## 设计

```python
# task_dw_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dwconv import DWConv


class TaskDWSelector(nn.Module):
    """
    Scale-wise Task-specific DWConv selector.

    Args:
        in_channels (int)
        num_tasks (int)
        return_weight (bool): 
            True -> return weight map
            False -> return reweighted feature
        detach_input (bool): whether to stop-grad encoder feature
    """

    def __init__(
        self,
        in_channels,
        num_tasks,
        return_weight=False,
        detach_input=False
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.return_weight = return_weight
        self.detach_input = detach_input

        self.task_dw = nn.ModuleList([
            DWConv(in_channels) for _ in range(num_tasks)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            weights or reweighted features
        """

        if self.detach_input:
            x = x.detach()

        outputs = []

        for dw in self.task_dw:
            weight = self.sigmoid(dw(x))

            if self.return_weight:
                outputs.append(weight)
            else:
                outputs.append(weight * x)

        return outputs  # list length = num_tasks
```

---

# 四、模块 3：TaskGating（方案二核心）

支持三种 gating 形式：

* conv 1x1
* MLP(GAP)
* MLP(GAP + task embedding)

统一接口。

---

## 设计

```python
# task_gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGating(nn.Module):
    """
    Generic Task Gating Module.

    mode:
        'conv'
        'mlp'
        'mlp_task_emb'
    """

    def __init__(
        self,
        in_channels,
        num_experts,
        num_tasks,
        mode='conv',
        hidden_dim=128
    ):
        super().__init__()

        self.mode = mode
        self.num_tasks = num_tasks
        self.num_experts = num_experts

        if mode == 'conv':
            self.gates = nn.ModuleList([
                nn.Conv2d(in_channels, num_experts, kernel_size=1)
                for _ in range(num_tasks)
            ])

        elif mode == 'mlp':
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_experts)
                )
                for _ in range(num_tasks)
            ])

        elif mode == 'mlp_task_emb':
            self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channels + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_experts)
                )
                for _ in range(num_tasks)
            ])

    def forward(self, x):
        """
        x: [B, C, H, W]
        return:
            list of gating weights (softmax over experts)
        """

        B, C, H, W = x.shape
        outputs = []

        for t in range(self.num_tasks):

            if self.mode == 'conv':
                g = self.gates[t](x)  # [B, K, H, W]

            else:
                gap = F.adaptive_avg_pool2d(x, 1).view(B, C)

                if self.mode == 'mlp':
                    g = self.gates[t](gap).unsqueeze(-1).unsqueeze(-1)

                else:
                    task_emb = self.task_embedding(
                        torch.full((B,), t, device=x.device)
                    )
                    inp = torch.cat([gap, task_emb], dim=1)
                    g = self.gates[t](inp).unsqueeze(-1).unsqueeze(-1)

            g = F.softmax(g, dim=1)

            outputs.append(g)

        return outputs
```

---

# 五、模块 4：FeatureAggregator（专家组合器）

用于：

> w = Σ α_k e_k

支持 spatial / non-spatial。

---

## 设计

```python
# feature_aggregator.py

import torch
import torch.nn as nn


class FeatureAggregator(nn.Module):
    """
    Combine expert outputs using gating weights.

    experts: list of expert modules
    """

    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, x, gating_weights):
        """
        x: [B, C, H, W]
        gating_weights: list length=num_tasks
                        each: [B, K, H, W] or [B, K, 1, 1]

        return:
            list of aggregated features
        """

        expert_outputs = [e(x) for e in self.experts]  # K

        expert_stack = torch.stack(expert_outputs, dim=1)
        # shape: [B, K, C, H, W]

        outputs = []

        for g in gating_weights:
            g = g.unsqueeze(2)  # [B, K, 1, H, W]
            weighted = (g * expert_stack).sum(dim=1)
            outputs.append(weighted)

        return outputs
```

---
你现在拥有：

* 方案一构件
* 方案二构件
* 方案三所需 gating 基础
* stop-grad 能力
* 可插拔专家结构

---
# 方案一实施方案
很好，现在我们进入 **E1（方案一）的工程落地规划阶段**。
目标非常明确：

> ✅ 实现 Scale-wise TaskDWSelector
> ✅ 插入 encoder → decoder 之间
> ✅ 不破坏现有训练流程
> ✅ 为 E4/E5（混合方案）和方案二预留接口

下面我按 **工程步骤 → 每步新增模块 → 预留扩展点 → 注意事项** 给你一个完整实施路径。

---

# 一、总体改造路径（非常重要）

你现在的结构大概率是：

```
x → encoder → {f1,f2,f3,f4,f5}
    → decoder_seg
    → decoder_sdf
    → decoder_bnd
```

我们要变成：

```
x → encoder → {f1,...,f5}
    → MultiScaleTaskSelector
        → {f1^(t),...,f5^(t)}
    → decoder_t
```

但要做到：

* Selector 是可开关的
* 默认行为 = 恒等映射
* 后续可以替换 f4/f5 为专家模块

---

# 二、E1 实现步骤规划（推荐按顺序）

---

# Step 1️⃣：创建 MultiScaleTaskSelector（总控模块）

这是整个方案一的核心。

## 目标

* 管理 f1–f5
* 每个尺度独立 TaskDWSelector
* 输出结构清晰
* 支持 scale-level 切换（为混合方案预留）

---

## 设计

```python
class MultiScaleTaskSelector(nn.Module):
    """
    Apply Task-wise DW feature selection on multi-scale features.

    Args:
        in_channels_list: list of channel numbers for f1-f5
        num_tasks: int
        mode: 'task_dw' | 'hybrid'
        hybrid_scales: list of scale indices using expert (e.g., [4,5])
    """

    def __init__(
        self,
        in_channels_list,
        num_tasks,
        mode='task_dw',
        hybrid_scales=None
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.mode = mode
        self.hybrid_scales = hybrid_scales or []

        self.selectors = nn.ModuleList()

        for i, c in enumerate(in_channels_list):
            scale_id = i + 1

            if mode == 'task_dw' or scale_id not in self.hybrid_scales:
                self.selectors.append(
                    TaskDWSelector(
                        in_channels=c,
                        num_tasks=num_tasks,
                        return_weight=False
                    )
                )
            else:
                # 预留专家结构（暂时不实现）
                self.selectors.append(None)

    def forward(self, features):
        """
        features: list [f1,...,f5]
        return:
            task_features: list length=num_tasks
                each element = list of 5 features
        """

        task_features = [
            [] for _ in range(self.num_tasks)
        ]

        for i, f in enumerate(features):

            selector = self.selectors[i]

            if selector is not None:
                outputs = selector(f)
            else:
                # placeholder (hybrid future)
                outputs = [f for _ in range(self.num_tasks)]

            for t in range(self.num_tasks):
                task_features[t].append(outputs[t])

        return task_features
```

---

## 这个模块的意义

它是：

> **E1 / E4 / E5 的统一入口**

后面：

* f4-f5 用专家？
* 只改这个模块
* decoder 不动
* encoder 不动

---

# Step 2️⃣：修改主模型 forward（轻改动）

假设原来：

```python
features = self.encoder(x)
out_seg = self.decoder_seg(features)
```

改为：

```python
features = self.encoder(x)

if self.use_feat_selector:
    task_features = self.selector(features)
else:
    task_features = [features] * self.num_tasks

out_seg = self.decoder_seg(task_features[0])
out_sdf = self.decoder_sdf(task_features[1])
out_bnd = self.decoder_bnd(task_features[2])
```

---

## 关键要求

* 不改 decoder 结构
* decoder 仍然接收 list[f1–f5]
* selector 开关可控

---

# Step 3️⃣：参数接口设计（必须做）

在 config 中加入：

```yaml
FEATURE_SELECT:
  ENABLE: true
  TYPE: task_dw
  HYBRID_SCALES: []
```

模型初始化：

```python
if cfg.FEATURE_SELECT.ENABLE:
    self.selector = MultiScaleTaskSelector(
        in_channels_list=[c1,c2,c3,c4,c5],
        num_tasks=3,
        mode=cfg.FEATURE_SELECT.TYPE,
        hybrid_scales=cfg.FEATURE_SELECT.HYBRID_SCALES
    )
```

---

# Step 4️⃣：为次生方案预留接口

你提出两个次生点：

---

## 次生方案 A

> f4/f5 用专家

已经在：

```
hybrid_scales=[4,5]
```

中预留。

未来只需：

```python
if scale_id in hybrid_scales:
    use ExpertSelector
```

---

## 次生方案 B

> 专家输出 + 原始特征共享

提前设计 FeatureAggregator 接口：

```python
aggregated = sum(alpha_k * expert_k)
if self.add_residual:
    aggregated = aggregated + x
```

建议在 ExpertSelector 内部留：

```python
self.use_residual = True
```

---

# Step 5️⃣：正则统计接口（现在就加）

非常关键。

在 TaskDWSelector 中加入：

```python
self.last_weight_maps = []
```

forward 中：

```python
self.last_weight_maps.append(weight)
```

未来：

* entropy loss
* 可视化
* collapse 检测

不用再改结构。

---

# 三、E1 完整改造流程总结

| 步骤 | 改动范围                      | 风险 |
| -- | ------------------------- | -- |
| 1  | 新建 MultiScaleTaskSelector | 无  |
| 2  | 主模型 forward 加 selector    | 低  |
| 3  | config 加开关                | 无  |
| 4  | 预留 hybrid_scales          | 无  |
| 5  | 正则统计接口                    | 无  |

---

# 四、非常关键的设计原则

现在做 E1 时必须满足：

* ❌ 不在 decoder 内写 gating
* ❌ 不改变 encoder
* ❌ 不 hardcode 任务数量
* ✅ selector 是 plug-in
* ✅ 可以一行关掉

---

# 五、E1 实现后的结构图（代码级）

```
Encoder
  ↓
[f1,f2,f3,f4,f5]
  ↓
MultiScaleTaskSelector
  ↓
[
  [f1_seg,...,f5_seg],
  [f1_sdf,...,f5_sdf],
  [f1_bnd,...,f5_bnd]
]
  ↓
Decoder_seg / sdf / bnd
```

---

# 六、下一步建议

现在最正确的推进方式：

1️⃣ 先只实现 E1
2️⃣ 跑通
3️⃣ 确认 loss 正常下降
4️⃣ 打印权重统计

然后我们再做：

* E2 专家模块插入
* 或 f4/f5 hybrid

---




