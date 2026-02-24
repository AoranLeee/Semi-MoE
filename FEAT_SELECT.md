很好。下面给出一次**结构闭合、目标明确、可直接执行的总修改方案**。
这是你当前实验阶段的**完整梯度更新与训练调度重构总设计**。

本次目标可以概括为：

> 在保持原始 **Semi-MoE** 思想的前提下，引入共享编码器 + MultiScale Selector，并重构优化器与梯度流控制，使训练过程稳定、可解释、可调试。

---

# 🔴 一、整体训练策略总原则

## 1️⃣ 梯度更新原则

| 阶段    | encoder | decoder | selector | gating | σ |
| ----- | ------- | ------- | -------- | ------ | - |
| sup   | ✅       | ✅       | ✅        | ✅      | ✅ |
| unsup | ✅       | ✅       | ✅        | ❌      | ✅ |

* 无 retain_graph
* unsup 阶段使用 `with torch.no_grad()` 冻结 gating
* 训练图分离（unsup 和 sup 分开 forward）

---

# 🔴 二、优化器最终结构（3 Optimizer）

## ✅ optimizer1（主优化器）

包含：

* backbone（encoder）
* decoder（三头）
* selector（DWConv × 15）

学习率结构：

* backbone lr = args.lr
* selector lr = args.lr × 5

实现方式（推荐 param_group，而非 add_param_group）：

```python
optimizer1 = torch.optim.SGD(
    [
        {"params": backbone_params, "lr": args.lr},
        {"params": decoder_params, "lr": args.lr},
        {"params": selector_params, "lr": args.lr * 5}
    ],
    momentum=args.momentum,
    weight_decay=5 * 10 ** args.wd
)
```

---

## ✅ optimizer2（MultiTaskLoss）

```python
optimizer2 = torch.optim.Adam(
    loss_fn.parameters(),
    lr=0.05,
    weight_decay=5 * 10 ** args.wd
)
```

---

## ✅ optimizer3（gating_model）

```python
optimizer3 = torch.optim.SGD(
    gating_model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=5 * 10 ** args.wd
)
```

---

# 🔴 三、Ramp-Up 策略设计

## 1️⃣ Selector α 曲线

* 类型：Sigmoid Ramp-up
* T = 20 epochs
* 作用：控制 selector 输出调制强度

### α(t) 公式：

[
\alpha(t) = \exp(-5(1 - t/T)^2)
]

* t ≥ T 时 α=1
* 前期接近 0
* 中期快速上升

### 每个 epoch 更新：

```python
alpha = sigmoid_rampup(epoch, 20)
selector.set_alpha(alpha)
```

---

## 2️⃣ Unsupervised Weight 曲线

* 类型：Sigmoid Ramp-up
* T = 80
* 不再使用线性增长

```python
unsup_weight = base_unsup_weight * sigmoid_rampup(epoch, 80)
```

---

## 3️⃣ LR Warmup

* warm_up_duration = 10 epochs
* 只作用于 optimizer1 和 optimizer3
* optimizer2 不 warmup

---

# 🔴 四、task_dw_selector.py 修改方案

## ✅ 1. 新增 α 参数

```python
self.alpha = 0.0
```

---

## ✅ 2. 前向中加入调制

```python
output = x * (1 + self.alpha * weight_map)
```

---

## ✅ 3. 新增 sigmoid ramp-up 函数

```python
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
```

---

## ✅ 4. 新增 get_weight_stats()

返回：

```python
{
    "mean": ...,
    "std": ...,
    "min": ...,
    "max": ...
}
```

统计当前 forward 产生的 weight maps。

---

# 🔴 五、multi_scale_task_selector.py 修改

新增：

## ✅ get_all_weight_stats()

返回：

```
{
  task1_scale1: {...},
  task1_scale2: {...},
  ...
  task3_scale5: {...}
}
```

包含所有 task × scale。

用于：

* 每5 epoch打印
* 写入日志文件

---

# 🔴 六、train.py 修改总任务

## 1️⃣ 优化器重构

* 删除 add_param_group
* 不再冻结 selector
* 不再 retain_graph

---

## 2️⃣ 训练循环结构

### 无监督阶段：

```python
optimizer1.zero_grad()
optimizer2.zero_grad()

with torch.no_grad():
    gating_out = gating_model(...)

loss_unsup.backward()

optimizer1.step()
optimizer2.step()
```

---

### 有监督阶段：

```python
optimizer1.zero_grad()
optimizer2.zero_grad()
optimizer3.zero_grad()

loss_sup.backward()

optimizer1.step()
optimizer2.step()
optimizer3.step()
```

---

## 3️⃣ 每 epoch 更新

```python
alpha = sigmoid_rampup(epoch, 20)
unsup_weight = base_weight * sigmoid_rampup(epoch, 80)
```

---

## 4️⃣ 每 5 epoch 打印：

* alpha
* unsup_weight
* selector weight mean
* selector weight std

---

## 5️⃣ 写入日志：

```
epoch, alpha, unsup_weight, selector_mean, selector_std
```

---

## 6️⃣ 参数统计增强

在参数统计部分加入：

```
Selector Params: xxx
Encoder Params: xxx
Decoder Params: xxx
Total Params: xxx
```

---

# 🔴 七、最终训练时间轴规划（200 epoch）

| 区间     | selector α | unsup_weight |
| ------ | ---------- | ------------ |
| 0–20   | 0 → 1      | 0 → 小        |
| 20–80  | 1          | 逐渐上升         |
| 80–200 | 1          | full         |

---

# 🔴 八、最终整体逻辑总结

## 训练稳定性保障：

* selector 不再冻结
* gating unsup 不更新
* 无 retain_graph
* 双 Sigmoid Ramp-up
* selector lr ×5
* warmup 10 epoch

---

# 🔴 九、本次修改完成后，你得到：

✅ 梯度更新完全可控
✅ 训练曲线可解释
✅ selector 不是随机噪声
✅ gating 不 self-confirm
✅ unsup 不会早期污染
✅ 实验可复现

---

# 🔴 十、本次任务本质

这不是简单调参。

这是：

> 将原始 Semi-MoE 多专家结构，重构为共享编码器 + 可学习特征调制机制，并保证梯度流逻辑等价。

---

