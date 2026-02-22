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

日志：
2.1.1增加打印参数量
新增 count_params 辅助函数。
在模型/优化器初始化后，打印一次总可训练参数量（M）。
在训练开始的第一个迭代中，重置并统计一次 GPU 峰值显存（GB），只打印一次。


2.2.1-2.2.8 增加共享编码器和CSV日志工具，记录各尺度数据，增加评判指标
1. unet.py
删除 SharedUNetMultiTask 和 unet_shared 工厂函数
保留并强化拆分：UNetEncoder / UNetDecoder
U_Net 现在由 UNetEncoder + UNetDecoder 组成，forward 行为保持一致
新增工厂函数：
create_unet_encoder(in_channels)
create_unet_decoder(num_classes)
encoder/decoder 都进行 kaiming 初始化
2. getnetwork.py
移除 unet_shared 分支，保留 unet 及其他网络不变
3. __init__.py
删除 unet_shared 导入
训练逻辑（核心改动）
4. train.py
新增 create_encoder / create_decoder（DDP 包装）
模型创建逻辑分支：
unet：三套完整模型
unet_shared：一个 shared_encoder + 三个 decoder（seg/sdf/bnd）
forward 路径分支：训练/验证都适配 shared encoder
优化器初始化适配 shared 结构（encoder + seg decoder 绑定在 optimizer1）
train/eval 模式切换适配 shared 结构
保存逻辑适配 shared 结构（checkpoint key 改为 shared_encoder/decoders）
total params 统计适配 shared 结构
5. 评估与日志统计
UNetEncoder 增加 last_features 缓存 + get_feature_stats()
train.py 增加 feature stats、seg entropy、encoder grad norm 记录
csv_logger.py
新增 CSVLogger 工具
CSV 日志集成（日志写入 {exp_name}.csv）
Seg Entropy 只打印平均值一次（不刷屏）
Debug 输出只打印一次：Encoder last_features updated: True/False
完成共享编码器实验、增加日志记录的semi-moe

2026.2.17 2.3.1增加特征融合基础模块
1) 训练逻辑（train.py）
删除原来create_encoder / create_decoder。
新增调用UNetMultiTask类，实现特征融合模块

2) U-Net 结构（unet.py）
新增 UNetMultiTask（3 个 decoder，输出 dict，支持 num_tasks==1 单任务回退）。
集成 MultiScaleTaskSelector（cfg.FEATURE_SELECT 控制），加入断言与 hybrid 警告。
forward 中新增 task_features 逻辑和一致性校验。

3) Feature Selector 模块（models/modules/feat_select/）
新增 dwconv.py（DWConv 模块）。
新增 task_dw_selector.py（TaskDWSelector，保存 last_weight_maps）。
更新 multi_scale_task_selector.py：
mode 默认 task_dw
hybrid_scales 逻辑
selector 插件点注释
输入/输出一致性检查

4) 网络注册
getnetwork.py 移除 unet_shared 分支。
__init__.py 移除 unet_shared 导入。
加入UNetMultiTask分支

5) 配置
新增 default.yaml，加入 FEATURE_SELECT 配置段（默认关闭）。
下一次加额外参数
