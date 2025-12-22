# Semi-MoE — 项目概览

此文档为第一次接触本项目的开发者准备，概述仓库的文件结构、每个主要部分的职责与快速上手提示（中文）。

仓库根（重要文件/目录）：

- `train.py` — 训练脚本（主入口）。包含模型实例化（多个模型：segment、sdf、boundary、gating）、数据加载、分布式初始化（`torch.distributed` + DDP）、优化器与 warmup 调度、训练/验证循环、保存 checkpoint 与结果。关键点：
  - 使用 DDP（单卡调试时脚本预设 `RANK=0`、`WORLD_SIZE=1` 并调用 `dist.init_process_group(backend='gloo', init_method='env://')`）。
  - 多模型并行训练：segment、sdf、boundary、gating，各自有 optimizer 与 scheduler。
  - gating 网络的输入是把三个 model 的 feature 在通道维拼接（`torch.cat([...], dim=1)`）。
  - weight decay 的计算为 `5 * 10 ** args.wd`（默认 `wd=-5` → 5e-5）。

- `test.py` — （测试/评估脚本）用于加载 checkpoint 并对验证/测试集做推理与评估（参见 README）。

- `README.md` — 项目说明与数据组织示例（包括 `train_sup_<N>`、`train_unsup_<N>`、`val` 文件夹结构）。

- `requirements.txt` — Python 依赖与版本锁定（如：torch==2.2.1、torchvision、timm、albumentations 等）。

- `aux_loss.py` — 项目中的若干辅助损失（例如 imbalance dice loss、sdf_loss、MultiTaskLoss 的实现或封装），训练流程复用这些实现。


核心目录与职责说明：

- `models/`
  - `getnetwork.py` — 名称到网络实现的映射工厂（`get_network(network, in_channels, num_classes, **kwargs)`）。新增模型需在这里注册。
  - `networks_2d/` — 多个 2D 网络实现（`unet.py`、`resunet.py`、`unet_plusplus.py`、`gating_net.py`、`hrnet.py` 等）。这些实现应满足前向契约：返回 `(features, logits)`。
  - `networks_3d/` — 3D 网络实现目录（如果你要做 3D 任务，模型放这里）。

- `dataload/`
  - `dataset_2d.py` — 2D 数据集与 DataLoader 构建（返回数据字段 keys：`image`, `mask`, `SDF`, `boundary`, `ID`）。训练脚本使用 `get_imagefolder` 等方法构造 `Dataset`。
  - `dataset_3d.py` — 3D 数据集实现（如果使用 3D 数据）。

- `config/`
  - `dataset_config/` (`dataset_cfg.py`) — 数据集相关常量（`IN_CHANNELS`, `NUM_CLASSES`, `PATH_DATASET`, `PATH_TRAINED_MODEL`, `PALETTE` 等），以及数据路径配置。修改/新增数据集时优先在此处添加配置。
  - `augmentation/` (`online_aug.py`) — 数据增强与归一化函数（训练/验证用不同 transform）。
  - `warmup_config/` (`warmup.py`) — GradualWarmupScheduler 的实现与用法（训练脚本中用来预热学习率）。
  - `train_test_config/` — 包含若干打印/保存/评估辅助函数（如 `print_train_loss`, `save_val_best_sup_2d` 等），用于训练与验证流程中标准化输出与模型保存。
  - `eval_config/` — 评估相关配置/脚本。

- `loss/`
  - `loss_function.py` — segmentation loss 的封装（例如 dice、cross-entropy 等组合），训练中作为 criterion 使用。


辅助与脚本：

- `aux_loss.py` — 辅助 loss 与 MultiTaskLoss 的实现（供训练脚本调用）。
- `test.py` — 测试脚本（加载 checkpoint 推理）。


项目关键约定（须遵守以避免行为偏差）：

1. 模型前向约定：大多数网络必须返回 `(features, logits)`。训练逻辑依赖 features 做 gating 输入，和 logits 做损失/输出。
2. 模型注册：向 `models/networks_2d/` 添加实现后，在 `models/getnetwork.py` 中注册字符串名以供 `train.py` 使用 `--network` / `--gating_network` 参数选择。
3. 数据目录命名：数据集根目录下应按 README 的示例放置 `train_sup_<N>`, `train_unsup_<N>`, `val`。`dataset_cfg.py` 保存路径常量。
4. 分布式：`train.py` 使用 `torch.distributed` + DDP，与分布式 sampler（`DistributedSampler`）与 `all_gather` 聚合结果；单卡调试可直接运行 `python train.py`（脚本自置环境变量）但真实多卡请用 `torchrun`/launcher 并设置相应 env。


快速上手与常用操作示例：

- 安装依赖：

```powershell
pip install -r requirements.txt
```

- 运行单卡训练（快速 smoke）：

```powershell
python train.py --num_epochs 1 --batch_size 1
```

（推荐先做单轮 smoke，确认数据路径正确、模型能前向并能保存 checkpoint。）

- 前向接口检查（交互式/调试）：

在 Python REPL 或小脚本中：

```python
from models.getnetwork import get_network
from config.dataset_config.dataset_cfg import dataset_cfg
cfg = dataset_cfg('CRAG')
net = get_network('unet', cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda()
feat, pred = net(torch.zeros(1, cfg['IN_CHANNELS'], 256, 256).cuda())
assert isinstance(feat, torch.Tensor) and isinstance(pred, torch.Tensor)
```


常见风险点与注意事项：

- weight decay 的计算使用 `5 * 10 ** args.wd`（不要随意改动）。
- 训练过程中存在多个模型和 optimizer，避免仅更新部分 optimizer 的 step/zero_grad，或在 refactor 时破坏梯度流。
- gating 网络通道大小依赖于各个模型返回的 feature channel 数（请确认通道数匹配）。


进一步建议（可选，后续实现）：

- 添加 `tests/smoke_forward.py` 来自动化前向契约检测（小脚本，CI 可运行）。
- 在 `CONTRIBUTING.md` 中补充“如何添加网络”和“如何添加数据集”的步骤（减少新贡献者失误）。


文件已生成：`PROJECT_OVERVIEW.md`（仓库根） — 如果你希望我把这份文档改成中英对照并列显示、或扩展为更详细的开发者 Guide（包含 DDP 调试范例与常见错误排查），告诉我我会继续完善。
