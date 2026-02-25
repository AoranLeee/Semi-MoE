import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.getnetwork import get_network
from models.networks_2d.unet import create_unet_encoder, create_unet_decoder, UNetMultiTask
import argparse
import time
import os
import numpy as np
import random
import yaml
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from dataload.dataset_2d import get_imagefolder
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss, print_val_loss, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_2d, print_best_sup
from warnings import simplefilter
from aux_loss import imbalance_diceLoss, sdf_loss, MultiTaskLoss
from csv_logger import CSVLogger

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)

# 忽略 FutureWarning（减少训练输出警告噪声）
simplefilter(action='ignore', category=FutureWarning)

# ------------------------------------------------------------------
# 以下环境变量用于通过 init_method='env://' 初始化 torch.distributed 时
# 提供最小的本地单卡调试配置：
# - RANK: 当前进程的分布式 rank（进程 id）。
# - WORLD_SIZE: 全部进程数量（总的 GPU/进程数）。
# - MASTER_ADDR / MASTER_PORT: 主节点地址和端口（env:// 需要这些信息以建立通信）。
#
# 在该项目中脚本将它们预置为单卡调试的默认值（RANK=0, WORLD_SIZE=1，MASTER 为本机）。
# 这对在没有 launcher（如 torchrun）的本地快速调试很方便，但在真实的多卡/多节点训练中：
#  - 请使用 `torchrun` / 集群 launcher 或外部脚本设置这些 env（不要在生产代码中硬编码）。
#  - 确保所用的 MASTER_PORT 在机器上未被占用。
# ------------------------------------------------------------------
os.environ["RANK"] = "0" #主进程
os.environ["WORLD_SIZE"] = "1" #单进程/单GPU
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "16672"
#定义一个工厂函数，用于根据字符串名称构建并封装网络
def create_model(network, in_channels, num_classes, **kwargs):
    #返回 (features, logits)
    model = get_network(network, in_channels, num_classes, **kwargs).cuda()
    return DistributedDataParallel(model, device_ids=[args.local_rank])

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def segmentation_entropy(logits, eps=1e-8):
    """
    logits: Tensor, shape [B, C, H, W]
    return: scalar entropy averaged over batch + spatial dims
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # [B, H, W]
    return entropy.mean()

def get_shared_encoder(model):
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "encoder", None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #选择数据集名称，GlasS或CRAG
    parser.add_argument('--dataset_name', default='CRAG', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--sup_mark', default='35')#用于拼接训练集目录名
    parser.add_argument('--unsup_mark', default='138')#按标注数量分割的数据集命名约定
    parser.add_argument('-b', '--batch_size', default=2, type=int)#批大小
    parser.add_argument('-e', '--num_epochs', default=200, type=int)#训练总 epoch 数
    parser.add_argument('-s', '--step_size', default=50, type=int)#学习率 StepLR 的步幅（每 step_size 个 epoch 乘以 gamma）
    parser.add_argument('-l', '--lr', default=0.5, type=float)#初始学习率。
    parser.add_argument('-g', '--gamma', default=0.5, type=float)#StepLR 的衰减因子：每过 step_size epoch，学习率乘以 gamma。
    parser.add_argument('-u', '--unsup_weight', default=0.2, type=float)#无监督部分 loss 的权重
    parser.add_argument('--loss', default='dice')#分割损失类型字符串，会用于构造 criterion（segmentation_loss(args.loss, False)）
    parser.add_argument('-w', '--warm_up_duration', default=20)#学习率预热的 epoch 数
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)#控制多少次迭代打印一次训练中间信息
    parser.add_argument('-n', '--network', default='unet_multitask', type=str)#主分割模型名称
    #Gating 网络名称
    parser.add_argument('-gn', '--gating_network', default='multi_gating_attention', type=str)
    parser.add_argument('--exp_name', default='exp_shared_encoder_baseline', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')#主进程的 rank，一般设为 0
    parser.add_argument('--visdom_port', default=16672)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    #初始化 PyTorch 分布式后端，建立进程间通信
    dist.init_process_group(backend='gloo', init_method='env://')

    rank = torch.distributed.get_rank()    #获取当前进程在全局通信中的 rank
    ngpus_per_node = torch.cuda.device_count()    #返回当前主机可见的 GPU 数量
    init_seeds(1)    #调用自定义的随机种子初始化函数

    logger = None
    if rank == args.rank_index:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = CSVLogger(os.path.join(log_dir, f"{args.exp_name}.csv"))

    dataset_name = args.dataset_name
    #调用配置函数 dataset_cfg（在 dataset_cfg.py 中）来加载与所选数据集相关的常量/路径字典
    cfg = dataset_cfg(dataset_name)#获取数据集配置字典

    with open('./config/feat_select_config/feat_select_cfg.yaml', 'r') as f:
        full_cfg = yaml.safe_load(f)
    feat_cfg = full_cfg['FEATURE_SELECT'] #获取特征选择相关配置字典

    print("feat_cfg:", feat_cfg)
    print("feat_cfg keys:", feat_cfg.keys())

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # trained model save
    #checkpoints/GlaS
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    if rank == args.rank_index:
        os.makedirs(path_trained_models, exist_ok=True)#创建主目录
    #checkpoints/GlaS/unet-l=0.01-e=50-s=20-g=0.5-b=2-uw=0.5-w=20-35-138
    path_trained_models = path_trained_models + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    if rank == args.rank_index:#创建子目录，存放训练模型
        os.makedirs(path_trained_models, exist_ok=True)

    # seg results save
    #seg_pred/GlaS
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if rank == args.rank_index:
        os.makedirs(path_seg_results, exist_ok=True)
    #seg_pred/GlaS/unet-l=0.01-e=50-s=20-g=0.5-b=2-uw=0.5-w=20-35-138
    path_seg_results = path_seg_results + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    if rank == args.rank_index:#创建子目录，存放分割结果
        os.makedirs(path_seg_results, exist_ok=True)

    data_transforms = data_transform_2d()
    #构建归一化（normalization）层/函数，把像素值标准化到训练所需分布
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_unsup = get_imagefolder( #创建“无监督训练集”（unsup）
        #无监督训练集的路径dataset/CRAG/train_unsup_138，从训练集138之后作为训练集
        data_dir=cfg['PATH_DATASET'] + '/train_unsup_' + args.unsup_mark,
        #无监督训练集的图像预处理
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=False,
        num_images=None,#不限制图像数量，使用目录中全部图片
    )#返回 Dataset 对象
    num_images_unsup = len(dataset_train_unsup)#无监督训练集的图像数量

    #dataset/CRAG/train_sup_35，从训练集前35张作为有监督训练集
    dataset_train_sup = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = get_imagefolder( #创建验证集
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )

    #创建训练集和验证集的采样器，shuffle=True 表示在每个 epoch 开始时会对索引打乱
    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup, shuffle=True)
    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    # 创建数据加载器
    dataloaders = dict() #初始化一个字典，用来保存三个 DataLoader（train_sup / train_unsup / val）
    #注意：这里 shuffle=False，因为 DistributedSampler 自己负责采样和打乱；如果同时给 shuffle=True 会冲突/无效。
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    #len(dataloader) 返回该 DataLoader 在当前设置下的迭代次数（即总样本数 / batch_size，受 sampler 影响）
    #计算并记录每个 DataLoader 在一个 epoch 中的批次数
    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    #输入通道数为 cfg['IN_CHANNELS']，输出通道为 cfg['NUM_CLASSES']（分割类别数）
    model = UNetMultiTask(
        in_channels=cfg['IN_CHANNELS'],
        num_classes=cfg['NUM_CLASSES'],
        cfg=feat_cfg
    ).cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    #输入的是第一层特征图，通道数为64；一共三个任务，所以乘3，和IN_CHANNELS其实没关系
    gating_model = create_model(args.gating_network, cfg['IN_CHANNELS'] * 64, cfg['NUM_CLASSES'])

    def _disable_inplace(m):
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False

    model.apply(_disable_inplace)
    gating_model.apply(_disable_inplace)

    dist.barrier()

    #构建分割损失实例（dice、ce、bce、bcebound）
    criterion = segmentation_loss(args.loss, False).cuda()
    #用于组合/加权多个子损失的包装器
    loss_fn = MultiTaskLoss().cuda()

    #为 model 创建一个 SGD 优化器
    model_for_opt = model.module if hasattr(model, "module") else model #检查 model 是否使用了分布式数据
    #获取 model 的编码器、解码器和选择模块的参数列表，准备传递给优化器
    backbone_params = (
        list(model_for_opt.encoder.parameters())
        + list(model_for_opt.decoder_seg.parameters())
        + list(model_for_opt.decoder_sdf.parameters())
        + list(model_for_opt.decoder_bnd.parameters())
    )
    selector_params = list(model_for_opt.selector.parameters()) if model_for_opt.selector is not None else []
    loss_params = list(loss_fn.parameters())
    gate_params = list(gating_model.parameters())

    base_lr = args.lr
    selector_lr = base_lr * feat_cfg['LR_MULTIPLIER']

    # IMPORTANT: keep routing-gate params (gate_params) separate from selector/adapter params.
    # - selector_params (selector + adapters) are feature modulation layers and must follow optimizer_main
    #   with their own LR (selector_lr).
    # - gate_params are Semi-MoE routing gates and must be optimized only by optimizer_gate.
    optimizer_main = optim.SGD(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": selector_params, "lr": selector_lr},
        ],
        momentum=args.momentum,
        weight_decay=5 * 10 ** args.wd
    )
    optimizer_gate = optim.SGD(
        gate_params,
        lr=base_lr,
        momentum=args.momentum,
        weight_decay=5 * 10 ** args.wd
    )
    optimizer_loss = optim.Adam(
        loss_params,
        lr=0.05,
        weight_decay=5 * 10 ** args.wd
    )
    
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_main, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer_main, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)
    
    exp_lr_scheduler4 = lr_scheduler.StepLR(
        optimizer_gate, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup4 = GradualWarmupScheduler(
        optimizer_gate, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler4)

    if rank == args.rank_index:
        total_params = (
            count_params(model)
            + count_params(gating_model)
            + count_params(loss_fn)
        )
        print(f"Total params: {total_params / 1e6:.2f} M")

    #记录训练开始时间，用于最后计算总耗时。
    since = time.time()
    #用作计数器（在每个 epoch 开始处递增），用于决定何时打印/记录信息
    count_iter = 0

    best_model = model
    best_result = 'Result1'
    #记录验证集上最佳评估指标的列表（初始化为 0）
    best_val_eval_list = [0 for i in range(4)]
    printed_memory = False
    printed_encoder_debug = False
    grad_enc_seg = None
    grad_enc_sdf = None
    grad_enc_bnd = None
    seg_entropy_sum = 0.0
    seg_entropy_count = 0
    feature_stats = {}
    selector_ramp_T = feat_cfg['ALPHA_RAMPUP_EPOCHS']
    unsup_ramp_T = feat_cfg['UNSUP_RAMPUP_EPOCHS']
    max_unsup_weight = args.unsup_weight

    for epoch in range(args.num_epochs):#200

        # ====== 逐步使用 selector ======
        selector_alpha = sigmoid_rampup(epoch, selector_ramp_T)
        unsup_weight = max_unsup_weight * sigmoid_rampup(epoch, unsup_ramp_T)
        model_for_alpha = model.module if hasattr(model, "module") else model
        model_for_alpha.set_selector_alpha(selector_alpha)
        if epoch % 5 == 0 and rank == args.rank_index:
            print(f"[Epoch {epoch}] selector_alpha={selector_alpha:.4f}, unsup_weight={unsup_weight:.4f}")

        count_iter += 1
        #每隔 display_iter（5）个 epoch 记录一次时间
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()
        if count_iter % args.display_iter == 0:
            grad_enc_seg = None
            grad_enc_sdf = None
            grad_enc_bnd = None
            seg_entropy_sum = 0.0
            seg_entropy_count = 0
            feature_stats = {}

        #设置 epoch 号，以便 DistributedSampler 在每个 epoch 开始时打乱数据
        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model.train()
        gating_model.train()

        #初始化累加的 epoch 损失值
        train_loss_sup_1 = 0.0 #有监督分割分支
        train_loss_sup_2 = 0.0 #有监督 SDF 分支
        train_loss_sup_3 = 0.0 #有监督边界分支
        train_loss_unsup = 0.0 #无监督部分
        train_loss = 0.0 #总损失

        val_loss_sup_1 = 0.0 #验证集有监督分割分支
        val_loss_sup_2 = 0.0 #验证集有监督 SDF 分支
        val_loss_sup_3 = 0.0 #验证集有监督边界分支

        #线性增加无监督 loss 的权重
        # unsup_weight set before batch loop (sigmoid ramp-up)
        dist.barrier() #同步所有进程

        #创建两个迭代器，分别用于无监督和有监督训练集，在 epoch 内通过 next() 按需取批
        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):
            if rank == args.rank_index and not printed_memory:
                torch.cuda.reset_peak_memory_stats()

            optimizer_main.zero_grad()
            optimizer_loss.zero_grad()
            optimizer_gate.zero_grad()

            #无监督---------------------------------------
            #从无监督 DataLoader 的迭代器中取下一批无标签数据
            unsup_index = next(dataset_train_unsup)
            img_train_unsup1 = unsup_index['image'].float().cuda()#取出图像张量并转换为 float，移动到当前 GPU
            #前向传播：通过三个模型分别计算特征feat和分割预测 logits的pred
            outputs = model(img_train_unsup1, detach_selector=True)
            feat_unsup1 = outputs["seg"][0]
            pred_train_unsup1 = outputs["seg"][1]
            feat_unsup2 = outputs["sdf"][0]
            pred_train_unsup2 = outputs["sdf"][1]
            feat_unsup3 = outputs["bnd"][0]
            pred_train_unsup3 = outputs["bnd"][1]

            #在 channel 维度上拼接三路特征，作为 gating 网络的输入
            gating_unsup_input = torch.cat([feat_unsup1, feat_unsup2, feat_unsup3], dim=1)
            #通过 gating 网络计算输出，融合后的分割 logits，SDF logits，边界 logits,用于生成伪标签
            unsup_out1, unsup_out2, unsup_out3 = gating_model(gating_unsup_input)

            #生成伪标签
            fake_bnd = torch.max(unsup_out3, dim=1)[1].detach()
            fake_sdf = (torch.tanh(unsup_out2)).detach() #tanh（将 SDF 值压缩到 [-1,1]）
            fake_mask = torch.max(unsup_out1, dim=1)[1].long().detach()
            
            #计算无监督损失
            loss_unsup_seg = criterion(pred_train_unsup1, fake_mask)
            loss_unsup_sdf = sdf_loss(torch.tanh(pred_train_unsup2), fake_sdf)
            loss_unsup_bnd = imbalance_diceLoss(pred_train_unsup3, fake_bnd)
            loss_train_unsup = loss_fn(loss_unsup_seg, loss_unsup_sdf, loss_unsup_bnd)

            loss_train_unsup = loss_train_unsup * unsup_weight
            # DDP note: unsup forward detaches selector, so some params are unused here.
            # Use no_sync() to avoid DDP reduction errors; sync happens on supervised backward.
            with model.no_sync():
                with gating_model.no_sync():
                    loss_train_unsup.backward()

            if rank == args.rank_index and not printed_memory:
                torch.cuda.synchronize()
                print(f"Memory Footprint (peak): {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                printed_memory = True

            #有监督---------------------------------------
            #从有监督 DataLoader 的迭代器中取下一批有标签数据
            sup_index = next(dataset_train_sup)
            img_train_sup1 = sup_index['image'].float().cuda()
            mask_train_sup = sup_index['mask'].cuda()
            sdf_train_sup = sup_index['SDF'].cuda()
            boundary_train_sup = sup_index['boundary'].cuda()
 
            #前向传播
            outputs = model(img_train_sup1, detach_selector=False)
            feat_sup1 = outputs["seg"][0]
            pred_train_sup1 = outputs["seg"][1]
            feat_sup2 = outputs["sdf"][0]
            pred_train_sup2 = outputs["sdf"][1]
            feat_sup3 = outputs["bnd"][0]
            pred_train_sup3 = outputs["bnd"][1]

            #拼接三路特征，作为 gating 网络的输入
            gating_sup_input = torch.cat([feat_sup1, feat_sup2, feat_sup3], dim=1)
            sup_out1, sup_out2, sup_out3 = gating_model(gating_sup_input)

            #记录训练集的分割预测和标签，用于计算评估指标
            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = sup_out1
                    mask_list_train = mask_train_sup
                    model_ref = model.module if hasattr(model, "module") else model
                    selector_stats = model_ref.get_selector_weight_stats()
                    if not isinstance(selector_stats, dict):
                        selector_stats = {}
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, sup_out1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            #计算有监督损失
            #把主分割模型的预测 pred_train_sup1 与 gating 的分割输出 sup_out1 都与真实 mask 比较，二者损失求和
            #损失函数不同
            loss_train_sup1 = (criterion(pred_train_sup1, mask_train_sup) + criterion(sup_out1, mask_train_sup))
            loss_train_sup2 = sdf_loss(torch.tanh(pred_train_sup2), sdf_train_sup) + sdf_loss(torch.tanh(sup_out2), sdf_train_sup) 
            loss_train_sup3 = imbalance_diceLoss(pred_train_sup3, boundary_train_sup) + imbalance_diceLoss(sup_out3, boundary_train_sup)

            #使用 MultiTaskLoss 将三个子损失组合成一个标量有监督损失
            loss_train_sup = loss_fn(loss_train_sup1, loss_train_sup2, loss_train_sup3)
            model_ref = model.module if hasattr(model, "module") else model
            loss_var = 0.0
            if hasattr(model_ref, "selector") and model_ref.selector is not None:
                loss_var = getattr(model_ref.selector, "last_loss_var", None)
                if loss_var is None:
                    loss_var = 0.0
            loss_sup_total = loss_train_sup + loss_var
            loss_sup_total.backward()
            loss_total = loss_train_sup + loss_train_unsup + loss_var

            #更新所有模型和 loss_fn 的参数
            optimizer_main.step()
            optimizer_loss.step()
            optimizer_gate.step()

            loss_train = loss_total #总损失
            train_loss_unsup += loss_train_unsup.item() #累加 epoch 累计值 ，用于统计打印
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss += loss_train.item()

        scheduler_warmup.step() #推进 optimizer 的学习率调度
        scheduler_warmup4.step()

        #验证集---------------------------------------
        #每隔 display_iter（5）个 epoch 在验证集上评估一次
        if count_iter % args.display_iter == 0:

            score_gather_list_train1 = [torch.zeros_like(score_list_train1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train1, score_list_train1)
            score_list_train1 = torch.cat(score_gather_list_train1, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
                train_eval_list1, train_m_jc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                encoder = model.module.encoder if hasattr(model, 'module') else model.encoder
                if encoder is not None and not printed_encoder_debug:
                    print(f"Encoder last_features updated: {encoder.last_features is not None}")
                    printed_encoder_debug = True
                feature_stats = encoder.get_feature_stats() if encoder is not None else {}
                enc_last_mean = feature_stats.get("enc_l5_mean", float("nan"))
                print(f"enc_l5_mean: {enc_last_mean:.6f}")
                model_ref = model.module if hasattr(model, "module") else model
                selector_stats_local = selector_stats if "selector_stats" in locals() and isinstance(selector_stats, dict) else {}
                print("[Selector Stats]")
                print(f"alpha: {selector_alpha:.4f}")
                print(f"unsup_weight: {unsup_weight:.4f}")
                if selector_stats_local:
                    for scale_key, stats in sorted(selector_stats_local.items(), key=lambda x: x[0]):
                        print(
                            "Scale {} -> mean={:.4f}".format(
                                scale_key,
                                stats.get("mean", float("nan")),
                            )
                        )
                else:
                    task_count = getattr(model_ref, "num_tasks", 0)
                    for idx in range(task_count):
                        print(
                            "Scale {} -> mean={:.4f}".format(
                                idx,
                                float("nan"),
                            )
                        )

                selector_var = float("nan")
                if hasattr(model_ref, "selector") and model_ref.selector is not None:
                    var_mean = getattr(model_ref.selector, "last_var_mean", None)
                    if var_mean is not None:
                        selector_var = var_mean.item()
                print(f"gate_variance: {selector_var:.6f}")
                def _scale_task_diff(scale_idx):
                    means = []
                    for task_idx in range(3):
                        key = f"scale{scale_idx}_task{task_idx}"
                        stats = selector_stats_local.get(key) if selector_stats_local else None
                        mean_val = stats.get("mean") if isinstance(stats, dict) else float("nan")
                        means.append(mean_val)
                    if all(isinstance(m, (int, float)) and not math.isnan(m) for m in means):
                        return max(means) - min(means)
                    return float("nan")
                diff0 = _scale_task_diff(0)
                diff4 = _scale_task_diff(4)
                print(f"scale0_task_diff: {diff0:.6f}")
                print(f"scale4_task_diff: {diff4:.6f}")
                # Print adapter beta_t for scale4 alongside scale4_task_diff
                beta_vals = []
                if hasattr(model_ref, "selector") and model_ref.selector is not None:
                    scale_selectors = getattr(model_ref.selector, "scale_selectors", None)
                    if isinstance(scale_selectors, list) and len(scale_selectors) > 4:
                        scale4_selector = scale_selectors[4]
                        if scale4_selector is not None and hasattr(scale4_selector, "adapter_beta"):
                            beta_vals = scale4_selector.adapter_beta.detach().cpu().tolist()
                if beta_vals:
                    beta_str = ", ".join(f"{b:.6f}" for b in beta_vals)
                    print(f"scale4_beta: [{beta_str}]")
                else:
                    print("scale4_beta: []")
            with torch.no_grad():
                model.eval()
                gating_model.eval()
                for i, data in enumerate(dataloaders['val']):

                    inputs_val1 = data['image'].float().cuda()
                    mask_val = data['mask'].cuda()
                    sdf_val = data['SDF'].cuda()
                    boundary_val = data['boundary'].cuda()
                    name_val = data['ID']

                    optimizer_main.zero_grad()
                    optimizer_loss.zero_grad()
                    optimizer_gate.zero_grad()
               
                    outputs = model(inputs_val1)
                    feat1 = outputs["seg"][0]
                    outputs_val1 = outputs["seg"][1]
                    feat2 = outputs["sdf"][0]
                    outputs_val2 = outputs["sdf"][1]
                    feat3 = outputs["bnd"][0]
                    outputs_val3 = outputs["bnd"][1]
                    gating_input = torch.cat([feat1, feat2, feat3], dim=1)
                    val_out1, val_out2, val_out3 = gating_model(gating_input)
                    if rank == args.rank_index:
                        seg_entropy = segmentation_entropy(outputs_val1)
                        seg_entropy_sum += seg_entropy.item()
                        seg_entropy_count += 1


                    if i == 0:
                        score_list_val1 = val_out1
                        mask_list_val = mask_val
                        name_list_val = name_val
                    else:
                        score_list_val1 = torch.cat((score_list_val1, val_out1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        name_list_val = np.append(name_list_val, name_val, axis=0)

                    loss_val_sup1 = criterion(outputs_val1, mask_val)
                    loss_val_sup2 = sdf_loss(torch.tanh(outputs_val2), sdf_val)
                    loss_val_sup3 = imbalance_diceLoss(outputs_val3, boundary_val)
                    val_loss_sup_1 += loss_val_sup1.item()
                    val_loss_sup_2 += loss_val_sup2.item()
                    val_loss_sup_3 += loss_val_sup3.item()

                score_gather_list_val1 = [torch.zeros_like(score_list_val1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val1, score_list_val1)
                score_list_val1 = torch.cat(score_gather_list_val1, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)

                name_gather_list_val = [None for _ in range(ngpus_per_node)]
                torch.distributed.all_gather_object(name_gather_list_val, name_list_val)
                name_list_val = np.concatenate(name_gather_list_val, axis=0)

                if rank == args.rank_index:
                    val_epoch_loss_sup1, val_epoch_loss_sup2, val_epoch_loss_sup3 = print_val_loss(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_minus)
                    val_eval_list1, val_m_jc1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val1, mask_list_val, print_num_minus)
                    if seg_entropy_count > 0:
                        seg_entropy_avg = seg_entropy_sum / seg_entropy_count
                    else:
                        seg_entropy_avg = float("nan")
                    print(f"Val Seg Entropy (avg): {seg_entropy_avg:.6f}")
                    save_models = {
                        'model': model,
                        'gating_model': gating_model
                    }
                    best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, save_models, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'MoE')

                    if logger is not None:
                        row = {"epoch": epoch + 1}
                        row["loss_seg"] = train_epoch_loss_sup1
                        row["loss_sdf"] = train_epoch_loss_sup2
                        row["loss_bnd"] = train_epoch_loss_sup3
                        row["loss_unsup"] = train_epoch_loss_unsup
                        row["seg_entropy"] = seg_entropy_avg
                        row["selector_alpha"] = selector_alpha
                        row["unsup_weight"] = unsup_weight
                        row["gate_variance"] = selector_var

                        def _get_scale_task_means(scale_idx):
                            means = []
                            for task_idx in range(3):
                                key = f"scale{scale_idx}_task{task_idx}"
                                stats = selector_stats.get(key) if isinstance(selector_stats, dict) else None
                                mean_val = stats.get("mean") if isinstance(stats, dict) else float("nan")
                                row[f"scale{scale_idx}_task{task_idx}_mean"] = mean_val
                                means.append(mean_val)
                            if all(isinstance(m, (int, float)) and not math.isnan(m) for m in means):
                                return max(means) - min(means)
                            return float("nan")

                        row["scale0_task_diff"] = _get_scale_task_means(0)
                        row["scale4_task_diff"] = _get_scale_task_means(4)
                        logger.log(row)

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(
                        print_num_minus, ' '), '|')

    #训练结束，打印总耗时和最佳验证集评估指标
    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
        print('=' * print_num)
