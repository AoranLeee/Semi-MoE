<!-- Copilot / AI agent instructions for the Semi-MoE repo -->
# Semi-MoE — AI coding agent guidance / 项目说明（中英双语）

This file is a short, actionable guide to help AI coding agents be productive in this repository.

本文件为 AI 编码代理（或协助者）准备的精简、可执行指南，帮助快速进入该仓库并做出安全修改。

Summary (one-line): Semi-MoE is a PyTorch-based semi-supervised Mixture-of-Experts segmentation codebase. Key areas: training loop (`train.py`), model factory (`models/getnetwork.py`), dataset config (`config/dataset_config/dataset_cfg.py`), data loaders (`dataload/dataset_2d.py`) and augmentations (`config/augmentation/online_aug.py`).

摘要（一行）：Semi-MoE 是基于 PyTorch 的半监督 Mixture-of-Experts 分割代码库。关键位置：训练脚本 `train.py`、模型工厂 `models/getnetwork.py`、数据集配置 `config/dataset_config/dataset_cfg.py`、数据加载 `dataload/dataset_2d.py` 和增强 `config/augmentation/online_aug.py`。

Quick run (what humans run):
- Install deps: `pip install -r requirements.txt`
- Train (single-node/one-GPU): `python train.py` (the script sets minimal `RANK`/`WORLD_SIZE` envs for single-GPU debugging)
- Test: `python test.py`

快速运行：
- 安装依赖：`pip install -r requirements.txt`
- 单机单卡训练（调试）：`python train.py`（脚本内部为单卡调试预置了 `RANK`/`WORLD_SIZE` 环境变量）
- 测试：`python test.py`

Important project contracts & patterns (use these when editing code):
- Model forward contract: most models return a tuple (features, logits). Example: `feat, pred = segment_model(inputs)` — many parts of the code concatenate features from three models and pass them to the gating network.
- Model selection: `models/getnetwork.py` contains the mapping from `--network` / `--gating_network` names to implementations. To add a network, add the implementation under `models/networks_2d/` (or `networks_3d/`) and register it in `getnetwork.py`.
- Dataset naming convention: datasets expect folders like `train_sup_<N>`, `train_unsup_<N>`, and `val` under the dataset root (see `README.md`). Configs for dataset paths and constants live in `config/dataset_config/dataset_cfg.py`.
- Distributed pattern: `train.py` uses `torch.distributed` + DDP. It uses distributed samplers and `all_gather` to aggregate scores across GPUs. For single-GPU debugging the script pre-sets `RANK=0` and `WORLD_SIZE=1`, and calls `dist.init_process_group(backend='gloo', init_method='env://')`.

重要约定与模式（修改代码时请遵循）：
- 模型前向约定：大多数网络返回 `(features, logits)` 二元组。例如：`feat, pred = segment_model(inputs)`。许多流程会把三个模型的 features 沿通道拼接后输入 gating 网络。
- 模型注册入口：`models/getnetwork.py` 管理 `--network` / `--gating_network` 字符串到实现的映射。新增网络时，请把实现放入 `models/networks_2d/`（或 `networks_3d/`）并在 `getnetwork.py` 注册。
- 数据集命名：数据集目录需要包含 `train_sup_<N>`、`train_unsup_<N>`、`val` 等（参见 `README.md`）。路径与常量位于 `config/dataset_config/dataset_cfg.py`。
- 分布式使用：`train.py` 使用 `torch.distributed` + DDP，训练中使用分布式 sampler 并通过 `all_gather` 聚合 score。单卡调试时脚本会预设 `RANK=0`、`WORLD_SIZE=1` 并调用 `dist.init_process_group(backend='gloo', init_method='env://')`。

Project-specific quirks to watch for:
- Weight decay is computed as `5 * 10 ** args.wd` in `train.py` (default `wd=-5` → effective weight_decay = 5e-5). Don't replace this without checking intent.
- Multiple models are trained in tandem (segment, sdf, boundary, gating). The training loop uses separate optimizers and warmup schedulers (`config/warmup_config/warmup.py`). Be careful when refactoring shared state.
- Gating network input: features from three parallel models are concatenated along channel dim before feeding the gating network (see gating input construction in `train.py`).
- Loss composition: a project-specific MultiTaskLoss and auxiliary losses live in `aux_loss.py` and `loss/loss_function.py`. Use existing loss wrappers rather than inlining new aggregation logic.

项目特有注意点：
- weight decay 的计算为 `5 * 10 ** args.wd`（默认 `wd=-5` → 有效 weight_decay = 5e-5）。不要随意改动，除非确认意图。
- 多个模型并行训练：segment、sdf、boundary、gating 分别有自己的 optimizer 和 warmup scheduler（参见 `config/warmup_config/warmup.py`）。重构共享状态时需谨慎。
- gating 网络输入：把三个模型的 feature 在通道维拼接后送入 gating（参见 `train.py` 中的 `torch.cat([...], dim=1)`）。
- 损失聚合：项目使用 `MultiTaskLoss` 和 `aux_loss.py` 中的辅助 loss。优先复用这些封装，避免在多个地方重复实现聚合逻辑。

Files worth reading for feature/context quickly:
- `train.py` — full training loop, distributed handling, model instantiation, major hyperparameters.
- `models/getnetwork.py` — mapping of network names to implementations.
- `dataload/dataset_2d.py` — dataset dataset contract (what keys the dataloader returns: `image`, `mask`, `SDF`, `boundary`, `ID`).
- `config/dataset_config/dataset_cfg.py` — dataset-specific constants like `IN_CHANNELS`, `NUM_CLASSES`, `PATH_DATASET`, `PATH_TRAINED_MODEL`, `PALETTE`.
- `config/augmentation/online_aug.py` — augmentation and normalization helpers used by `train.py`.
- `aux_loss.py`, `loss/loss_function.py` — project loss implementations and composer.

快速查看的文件：
- `train.py` — 完整训练循环、分布式处理、模型实例化与关键超参。
- `models/getnetwork.py` — 模型名称与实现的映射。
- `dataload/dataset_2d.py` — dataloader 的 contract（返回的 key：`image`, `mask`, `SDF`, `boundary`, `ID`）。
- `config/dataset_config/dataset_cfg.py` — 数据集相关常量：`IN_CHANNELS`, `NUM_CLASSES`, `PATH_DATASET`, `PATH_TRAINED_MODEL`, `PALETTE` 等。
- `config/augmentation/online_aug.py` — 增强与归一化工具，训练脚本使用它们构建 transforms。
- `aux_loss.py`, `loss/loss_function.py` — 损失实现与聚合器。

Examples for common tasks (concrete):
- Switch to a different encoder: run `python train.py --network unet_plusplus` and ensure `getnetwork.py` supports the name.
- Add a new dataset: update `config/dataset_config/dataset_cfg.py` with path and dataset constants, place data in `PATH_DATASET/train_sup_<N>` and `PATH_DATASET/train_unsup_<N>`.

常见任务示例：
- 切换网络：`python train.py --network unet_plusplus`，并在 `models/getnetwork.py` 注册对应名称。
- 新增数据集：修改 `config/dataset_config/dataset_cfg.py` 中的路径/常量，数据放到 `PATH_DATASET/train_sup_<N>` 与 `PATH_DATASET/train_unsup_<N>`。

When changing behavior, run these quick checks locally:
- Single-epoch smoke: reduce `--num_epochs 1 --batch_size 1` and run `python train.py` to ensure the loop runs and saves a checkpoint.
- Forward contract check: call `feat, pred = net(torch.zeros(1, cfg['IN_CHANNELS'], H, W).cuda())` to verify network returns (feat, pred).

修改验证建议：
- 单轮快速检测：`--num_epochs 1 --batch_size 1` 运行 `python train.py`，验证训练循环与 checkpoint 存储能够正常执行。
- 前向接口检测：运行 `feat, pred = net(torch.zeros(1, cfg['IN_CHANNELS'], H, W).cuda())`，确认网络返回 `(feat, pred)`。

If you need clarification, ask about: where dataset paths are configured (the dataset_cfg), which network name to register, or whether new loss components should integrate into `MultiTaskLoss`.

如需澄清，请说明：需要确认的数据集路径（`dataset_cfg`）、应注册的网络名称，或新增损失是否应并入 `MultiTaskLoss`。

This file is intentionally short; if you want more detail (examples for debugging DDP, tests to add, or a guided walkthrough of the training loop), say which area to expand.

本说明较短；如需更详细内容（DDP 调试示例、自动化 smoke tests、训练循环逐行讲解等），请告知具体需求。
