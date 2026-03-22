import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.getnetwork import get_network
import argparse
import time
import os
import numpy as np
import random
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from loss.loss_function import segmentation_loss
from dataload.dataset_2d import get_imagefolder
from config.warmup_config.warmup import GradualWarmupScheduler
from config.train_test_config.train_test_config import print_train_loss, print_val_loss, print_val_eval_sup, save_val_best_sup_2d, print_best_sup
from config.eval_config.eval import evaluate, evaluate_multi
from warnings import simplefilter
from aux_loss import imbalance_diceLoss, sdf_loss, MultiTaskLoss
from csv_logger import CSVLogger
from models.modules.uncertainty.uncertainty import (
    symmetric_kl_uncertainty,
    sdf_uncertainty
)
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)

def get_unsup_weight(epoch, unsup_warmup_type, max_unsup_weight, num_epochs, rampup_length):
    if unsup_warmup_type == 'linear':
        return max_unsup_weight * float(epoch + 1) / float(max(1, num_epochs))
    if unsup_warmup_type == 'sigmoid':
        return max_unsup_weight * sigmoid_rampup(epoch, rampup_length)
    raise ValueError(f"Unsupported unsup warmup type: {unsup_warmup_type}")


# 忽略 FutureWarning（减少训练输出警告噪声）
simplefilter(action='ignore', category=FutureWarning)

os.environ["RANK"] = "0" 
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "16672"
#定义一个工厂函数，用于根据字符串名称构建并封装网络
class MultiTaskOutputAdapter(nn.Module):
    """
    Unified output:
      outputs["seg"], outputs["sdf"], outputs["bnd"]
    where each value is (feature, logits).
    """
    def __init__(self, model, network):
        super().__init__()
        self.model = model
        self.network = network
    @property
    def encoder(self):
        return getattr(self.model, "encoder", None)

    @property
    def decoder_seg(self):
        return getattr(self.model, "decoder_seg", None)

    @property
    def decoder_sdf(self):
        return getattr(self.model, "decoder_sdf", None)

    @property
    def decoder_bnd(self):
        return getattr(self.model, "decoder_bnd", None)

    @property
    def decoder(self):
        return getattr(self.model, "decoder", None)



    def _normalize_outputs(self, outputs):
        if isinstance(outputs, dict) and all(k in outputs for k in ("seg", "sdf", "bnd")):
            return outputs

        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            feat, seg_logits = outputs
            if seg_logits.dim() < 4:
                raise ValueError("Expected seg logits shape [B, C, H, W].")
            return {
                "seg": (feat, seg_logits),
                "sdf": (feat, seg_logits[:, :1, ...]),
                "bnd": (feat, seg_logits),
            }

        raise TypeError(f"Unsupported output type for network '{self.network}': {type(outputs)}")
    def forward(self, x):
        outputs = self.model(x)
        return self._normalize_outputs(outputs)


def create_model(network, in_channels, num_classes, local_rank, normalize_outputs=False, **kwargs):
    model = get_network(
        network=network,
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )
    if normalize_outputs:
        model = MultiTaskOutputAdapter(model, network=network)
    model = model.cuda()
    return DistributedDataParallel(model, device_ids=[local_rank])

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


def print_train_core_metrics(num_classes, score_list_train, mask_list_train, print_num_minus):
    if num_classes == 2:
        eval_list = evaluate(score_list_train, mask_list_train)
        train_thr = float(eval_list[0])
        train_jc = float(eval_list[1])
        train_dc = float(eval_list[2])
    else:
        eval_list = evaluate_multi(score_list_train, mask_list_train)
        train_thr = float("nan")
        train_jc = float(eval_list[1])
        train_dc = float(eval_list[3])

    print('| Train Thr: {:.4f}'.format(train_thr).ljust(print_num_minus, ' '), '|')
    print('| Train  Jc: {:.4f}'.format(train_jc).ljust(print_num_minus, ' '), '|')
    print('| Train  Dc: {:.4f}'.format(train_dc).ljust(print_num_minus, ' '), '|')

    return train_thr, train_jc, train_dc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #选择数据集名称，GlasS或CRAG
    parser.add_argument('--dataset_name', default='CRAG', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--sup_mark', default='35')#用于拼接训练集目录名
    parser.add_argument('--unsup_mark', default='138')#按标注数量分割的数据集命名约�?
    parser.add_argument('-b', '--batch_size', default=2, type=int)#批大�?
    parser.add_argument('-e', '--num_epochs', default=200, type=int)#训练�?epoch �?
    parser.add_argument('-s', '--step_size', default=50, type=int)#学习�?StepLR 的步幅（�?step_size �?epoch 乘以 gamma�?
    parser.add_argument('-l', '--lr', default=0.5, type=float)#初始学习率�?
    parser.add_argument('-g', '--gamma', default=0.5, type=float)#StepLR 的衰减因子：每过 step_size epoch，学习率乘以 gamma�?
    parser.add_argument('-u', '--unsup_weight', default=0.5, type=float)#无监督部�?loss 的权�?
    parser.add_argument('--lambda_u', default=0.0, type=float)
    parser.add_argument('--unsup_warmup_type',default='linear',choices=['linear', 'sigmoid'])
    parser.add_argument('--loss', default='dice')#分割损失类型字符串，会用于构�?criterion（segmentation_loss(args.loss, False)�?
    parser.add_argument('-w', '--warm_up_duration', default=20)#学习率预热的 epoch �?
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)#控制多少次迭代打印一次训练中间信�?
    parser.add_argument('-n', '--network', default='unet_shared', choices=['unet_shared'], type=str)
    #Gating 网络名称
    parser.add_argument('-gn', '--gating_network', default='multi_gating_attention', type=str)
    parser.add_argument('--exp_name', default='exp_shared_encoder_baseline', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')#主进程的 rank，一般设�?0
    parser.add_argument('--visdom_port', default=16672)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')

    rank = torch.distributed.get_rank()    #获取当前进程在全局通信中的 rank
    ngpus_per_node = torch.cuda.device_count()    #返回当前主机可见�?GPU 数量
    init_seeds(1)    #调用自定义的随机种子初始化函�?

    logger = None
    if rank == args.rank_index:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = CSVLogger(os.path.join(log_dir, f"{args.exp_name}.csv"))

    dataset_name = args.dataset_name
    #调用配置函数 dataset_cfg（在 dataset_cfg.py 中）来加载与所选数据集相关的常�?路径字典
    cfg = dataset_cfg(dataset_name)#获取数据集配置字�?
    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2

    # trained model save
    #checkpoints/GlaS
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    if rank == args.rank_index:
        os.makedirs(path_trained_models, exist_ok=True)#创建主目�?
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

    dataset_train_unsup = get_imagefolder( #创建“无监督训练集”（unsup�?
        #无监督训练集的路径dataset/CRAG/train_unsup_138，从训练�?38之后作为训练�?
        data_dir=cfg['PATH_DATASET'] + '/train_unsup_' + args.unsup_mark,
        #无监督训练集的图像预处理
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=False,
        num_images=None,#不限制图像数量，使用目录中全部图�?
    )#返回 Dataset 对象
    num_images_unsup = len(dataset_train_unsup)#无监督训练集的图像数�?

    #dataset/CRAG/train_sup_35，从训练集前35张作为有监督训练�?
    dataset_train_sup = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = get_imagefolder( #创建验证�?
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )

    #创建训练集和验证集的采样器，shuffle=True 表示在每�?epoch 开始时会对索引打乱
    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup, shuffle=True)
    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    # 创建数据加载�?
    dataloaders = dict() #初始化一个字典，用来保存三个 DataLoader（train_sup / train_unsup / val�?
    #注意：这�?shuffle=False，因�?DistributedSampler 自己负责采样和打乱；如果同时�?shuffle=True 会冲�?无效�?
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    #len(dataloader) 返回�?DataLoader 在当前设置下的迭代次数（即总样本数 / batch_size，受 sampler 影响�?
    #计算并记录每�?DataLoader 在一�?epoch 中的批次�?
    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    #输入通道数为 cfg['IN_CHANNELS']，输出通道�?cfg['NUM_CLASSES']（分割类别数�?
    model = create_model(
        network=args.network,
        in_channels=cfg['IN_CHANNELS'],
        num_classes=cfg['NUM_CLASSES'],
        local_rank=args.local_rank,
        normalize_outputs=True,
    )
    #输入的是第一层特征图，通道数为64；一共三个任务，所以乘3，和IN_CHANNELS其实没关�?
    gating_model = create_model(
        network=args.gating_network,
        in_channels=cfg['IN_CHANNELS'] * 64,
        num_classes=cfg['NUM_CLASSES'],
        local_rank=args.local_rank,
    )

    def _disable_inplace(m):
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False

    model.apply(_disable_inplace)
    gating_model.apply(_disable_inplace)

    dist.barrier()

    #构建分割损失实例（dice、ce、bce、bcebound�?
    criterion = segmentation_loss(args.loss, False).cuda()
    #用于组合/加权多个子损失的包装�?
    loss_fn = MultiTaskLoss().cuda()

    #�?model 创建一�?SGD 优化�?
    model_for_opt = model.module if hasattr(model, "module") else model #检�?model 是否使用了分布式数据
    #获取 model 的编码器、解码器和选择模块的参数列表，准备传递给优化�?
    if (
        model_for_opt.encoder is not None
        and model_for_opt.decoder_seg is not None
        and model_for_opt.decoder_sdf is not None
        and model_for_opt.decoder_bnd is not None
    ):
        backbone_params = (
            list(model_for_opt.encoder.parameters())
            + list(model_for_opt.decoder_seg.parameters())
            + list(model_for_opt.decoder_sdf.parameters())
            + list(model_for_opt.decoder_bnd.parameters())
        )
    elif model_for_opt.encoder is not None and model_for_opt.decoder is not None:
        backbone_params = list(model_for_opt.encoder.parameters()) + list(model_for_opt.decoder.parameters())
    else:
        backbone_params = list(model_for_opt.parameters())

    loss_params = list(loss_fn.parameters())
    gate_params = list(gating_model.parameters())

    base_lr = args.lr

    # - gate_params are Semi-MoE routing gates and must be optimized only by optimizer_gate.
    optimizer_main = optim.SGD(
        backbone_params,
        lr=base_lr,
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

    printed_memory = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.local_rank)

    if rank == args.rank_index:
        total_params = (
            count_params(model)
            + count_params(gating_model)
            + count_params(loss_fn)
        )
        print(f"Total params: {total_params / 1e6:.2f} M")

    #记录训练开始时间，用于最后计算总耗时�?
    since = time.time()
    #用作计数器（在每�?epoch 开始处递增），用于决定何时打印/记录信息
    count_iter = 0

    #记录验证集上最佳评估指标的列表（初始化�?0�?
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):#200
        unsup_weight = get_unsup_weight(
            epoch=epoch,
            unsup_warmup_type=args.unsup_warmup_type,
            max_unsup_weight=args.unsup_weight,
            num_epochs=args.num_epochs,
            rampup_length=80,
        )

        count_iter += 1
        #每隔 display_iter）个 epoch 记录一次时�?
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        #设置 epoch 号，以便 DistributedSampler 在每�?epoch 开始时打乱数据
        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model.train()
        gating_model.train()

        #初始化累加的 epoch 损失�?
        train_loss_sup_1 = 0.0 #有监督分割分�?
        train_loss_sup_2 = 0.0 #有监�?SDF 分支
        train_loss_sup_3 = 0.0 #有监督边界分�?
        train_loss_unsup = 0.0 #无监督部�?
        train_loss = 0.0 #总损�?

        val_loss_sup_1 = 0.0 #验证集有监督分割分支
        val_loss_sup_2 = 0.0 #验证集有监督 SDF 分支
        val_loss_sup_3 = 0.0 #验证集有监督边界分支

        # ===== Uncertainty logging =====
        uncert_stats = {
            "U_seg": 0.0, "U_sdf": 0.0, "U_bnd": 0.0,
            "W_seg": 0.0, "W_sdf": 0.0, "W_bnd": 0.0,
            "raw_seg": 0.0, "raw_sdf": 0.0, "raw_bnd": 0.0,
            "weighted_seg": 0.0, "weighted_sdf": 0.0, "weighted_bnd": 0.0,
            "count": 0
        }

        uncert_seg_dist = {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0
        }
        #线性增加无监督 loss 的权�?
        # unsup_weight set before batch loop (sigmoid ramp-up)
        dist.barrier() #同步所有进�?

        #创建两个迭代器，分别用于无监督和有监督训练集，在 epoch 内通过 next() 按需取批
        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):
            optimizer_main.zero_grad()
            optimizer_loss.zero_grad()
            optimizer_gate.zero_grad()

            #无监�?--------------------------------------
            #从无监督 DataLoader 的迭代器中取下一批无标签数据
            unsup_index = next(dataset_train_unsup)
            img_train_unsup1 = unsup_index['image'].float().cuda()#取出图像张量并转换为 float，移动到当前 GPU
            #前向传播：通过三个模型分别计算特征feat和分割预�?logits的pred
            outputs = model(img_train_unsup1)
            feat_unsup1 = outputs["seg"][0]
            pred_train_unsup1 = outputs["seg"][1]
            feat_unsup2 = outputs["sdf"][0]
            pred_train_unsup2 = outputs["sdf"][1]
            feat_unsup3 = outputs["bnd"][0]
            pred_train_unsup3 = outputs["bnd"][1]

            #�?channel 维度上拼接三路特征，作为 gating 网络的输�?
            gating_unsup_input = torch.cat([
                feat_unsup1.detach(),
                feat_unsup2.detach(),
                feat_unsup3.detach()
            ], dim=1)
            #通过 gating 网络计算输出，融合后的分�?logits，SDF logits，边�?logits,用于生成伪标�?
            unsup_out1, unsup_out2, unsup_out3 = gating_model(gating_unsup_input)

            #计算无监督损�?
            # ======================
            # Uncertainty computation (UNSUP ONLY, pixel-wise)
            # ======================
            U_seg = symmetric_kl_uncertainty(pred_train_unsup1, unsup_out1.detach())  # (B,H,W)
            U_sdf = sdf_uncertainty(pred_train_unsup2, unsup_out2.detach())            # (B,H,W)
            U_bnd = symmetric_kl_uncertainty(pred_train_unsup3, unsup_out3.detach())   # (B,H,W)

            # 1. pseudo label from teacher
            pseudo_seg = torch.argmax(unsup_out1.detach(), dim=1)  # (B,H,W)
            pseudo_bnd = torch.argmax(unsup_out3.detach(), dim=1)

            # 2. pixel-wise CE
            loss_map_seg = F.cross_entropy(pred_train_unsup1, pseudo_seg, reduction='none')
            loss_map_bnd = F.cross_entropy(pred_train_unsup3, pseudo_bnd, reduction='none')

            # 3. pixel-wise SDF loss
            loss_map_sdf = (torch.tanh(pred_train_unsup2) - torch.tanh(unsup_out2.detach())) ** 2
            loss_map_sdf = loss_map_sdf.squeeze(1)

            # raw (unweighted) pixel loss for logging
            loss_unsup_seg_raw = loss_map_seg.mean()
            loss_unsup_sdf_raw = loss_map_sdf.mean()
            loss_unsup_bnd_raw = loss_map_bnd.mean()

            # 4. uncertainty weighting (pixel-wise)
            W_seg = 1.0 / (1.0 + U_seg)
            W_sdf = 1.0 / (1.0 + U_sdf)
            W_bnd = 1.0 / (1.0 + U_bnd)

            loss_unsup_seg = (W_seg * loss_map_seg).mean()
            loss_unsup_sdf = (W_sdf * loss_map_sdf).mean()
            loss_unsup_bnd = (W_bnd * loss_map_bnd).mean()

            # 5. uncertainty regularization
            lambda_u = args.lambda_u
            loss_uncert_reg = (
                U_seg.mean() +
                U_sdf.mean() +
                U_bnd.mean()
            )

            with torch.no_grad():
                # weights
                w_seg = W_seg.mean()
                w_sdf = W_sdf.mean()
                w_bnd = W_bnd.mean()

                # accumulate
                uncert_stats["U_seg"] += U_seg.mean().item()
                uncert_stats["U_sdf"] += U_sdf.mean().item()
                uncert_stats["U_bnd"] += U_bnd.mean().item()

                uncert_stats["W_seg"] += w_seg.item()
                uncert_stats["W_sdf"] += w_sdf.item()
                uncert_stats["W_bnd"] += w_bnd.item()

                uncert_stats["raw_seg"] += loss_unsup_seg_raw.item()
                uncert_stats["raw_sdf"] += loss_unsup_sdf_raw.item()
                uncert_stats["raw_bnd"] += loss_unsup_bnd_raw.item()

                uncert_stats["weighted_seg"] += loss_unsup_seg.item()
                uncert_stats["weighted_sdf"] += loss_unsup_sdf.item()
                uncert_stats["weighted_bnd"] += loss_unsup_bnd.item()

                uncert_stats["count"] += 1

                # seg uncertainty distribution
                uncert_seg_dist["mean"] += U_seg.mean().item()
                uncert_seg_dist["max"] += U_seg.max().item()
                uncert_seg_dist["min"] += U_seg.min().item()

            # 6. multitask weighting + uncertainty regularization
            loss_train_unsup = loss_fn(loss_unsup_seg, loss_unsup_sdf, loss_unsup_bnd) + lambda_u * loss_uncert_reg
            loss_train_unsup = loss_train_unsup * unsup_weight
            with model.no_sync():
                with gating_model.no_sync():
                    loss_train_unsup.backward()

            #有监�?--------------------------------------
            #从有监督 DataLoader 的迭代器中取下一批有标签数据
            sup_index = next(dataset_train_sup)
            img_train_sup1 = sup_index['image'].float().cuda()
            mask_train_sup = sup_index['mask'].cuda()
            sdf_train_sup = sup_index['SDF'].cuda()
            boundary_train_sup = sup_index['boundary'].cuda()
 
            #前向传播
            outputs = model(img_train_sup1)
            feat_sup1 = outputs["seg"][0]
            pred_train_sup1 = outputs["seg"][1]
            feat_sup2 = outputs["sdf"][0]
            pred_train_sup2 = outputs["sdf"][1]
            feat_sup3 = outputs["bnd"][0]
            pred_train_sup3 = outputs["bnd"][1]

            #拼接三路特征，作�?gating 网络的输�?
            gating_sup_input = torch.cat([
                feat_sup1.detach(),
                feat_sup2.detach(),
                feat_sup3.detach()
            ], dim=1)
            sup_out1, sup_out2, sup_out3 = gating_model(gating_sup_input)

            #记录训练集的分割预测和标签，用于计算评估指标
            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = sup_out1
                    mask_list_train = mask_train_sup
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, sup_out1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            #计算有监督损�?
            #把主分割模型的预�?pred_train_sup1 �?gating 的分割输�?sup_out1 都与真实 mask 比较，二者损失求�?
            #损失函数不同
            loss_train_sup1 = (criterion(pred_train_sup1, mask_train_sup) + criterion(sup_out1, mask_train_sup))
            loss_train_sup2 = sdf_loss(torch.tanh(pred_train_sup2), sdf_train_sup) + sdf_loss(torch.tanh(sup_out2), sdf_train_sup) 
            loss_train_sup3 = imbalance_diceLoss(pred_train_sup3, boundary_train_sup) + imbalance_diceLoss(sup_out3, boundary_train_sup)

            loss_train_sup = loss_fn(loss_train_sup1, loss_train_sup2, loss_train_sup3)
            loss_train_sup.backward()
            loss_total = loss_train_sup + loss_train_unsup

            #更新所有模型和 loss_fn 的参�?
            optimizer_main.step()
            optimizer_loss.step()
            optimizer_gate.step()

            if rank == args.rank_index and not printed_memory:
                torch.cuda.synchronize(args.local_rank)
                peak_alloc_gb = torch.cuda.max_memory_allocated(args.local_rank) / (1024 ** 3)
                print(f"GPU Peak Memory (peak allocated): {peak_alloc_gb:.2f} GB")
                printed_memory = True

            loss_train = loss_total #总损�?
            train_loss_unsup += loss_train_unsup.item() #累加 epoch 累计�?，用于统计打�?
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss += loss_train.item()

        scheduler_warmup.step() #推进 optimizer 的学习率调度
        scheduler_warmup4.step()

        #验证�?--------------------------------------
        #每隔 display_iter�?）个 epoch 在验证集上评估一�?
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
                train_thr, train_jc, train_dc = print_train_core_metrics(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                print('| unsup_weight: {:.4f}, warmup_type: {}'.format(unsup_weight, args.unsup_warmup_type).ljust(print_num_minus, ' '), '|')
                # ===== Uncertainty print =====
                if uncert_stats["count"] > 0:
                    c = uncert_stats["count"]

                    U_seg_avg = uncert_stats["U_seg"] / c
                    U_sdf_avg = uncert_stats["U_sdf"] / c
                    U_bnd_avg = uncert_stats["U_bnd"] / c

                    W_seg_avg = uncert_stats["W_seg"] / c
                    W_sdf_avg = uncert_stats["W_sdf"] / c
                    W_bnd_avg = uncert_stats["W_bnd"] / c

                    raw_seg_avg = uncert_stats["raw_seg"] / c
                    raw_sdf_avg = uncert_stats["raw_sdf"] / c
                    raw_bnd_avg = uncert_stats["raw_bnd"] / c

                    weighted_seg_avg = uncert_stats["weighted_seg"] / c
                    weighted_sdf_avg = uncert_stats["weighted_sdf"] / c
                    weighted_bnd_avg = uncert_stats["weighted_bnd"] / c

                    U_seg_mean = uncert_seg_dist["mean"] / c
                    U_seg_max = uncert_seg_dist["max"] / c
                    U_seg_min = uncert_seg_dist["min"] / c

                    print('| Uncertainty Stats '.ljust(print_num_minus, ' '), '|')
                    print(f'| U(seg/sdf/bnd): {U_seg_avg:.4f} / {U_sdf_avg:.4f} / {U_bnd_avg:.4f}'.ljust(print_num_minus, ' '), '|')
                    print(f'| W(seg/sdf/bnd): {W_seg_avg:.4f} / {W_sdf_avg:.4f} / {W_bnd_avg:.4f}'.ljust(print_num_minus, ' '), '|')
                    print(f'| RawLoss(seg/sdf/bnd): {raw_seg_avg:.4f} / {raw_sdf_avg:.4f} / {raw_bnd_avg:.4f}'.ljust(print_num_minus, ' '), '|')
                    print(f'| WeightedLoss(seg/sdf/bnd): {weighted_seg_avg:.4f} / {weighted_sdf_avg:.4f} / {weighted_bnd_avg:.4f}'.ljust(print_num_minus, ' '), '|')
                    print(f'| U_seg dist (mean/max/min): {U_seg_mean:.4f} / {U_seg_max:.4f} / {U_seg_min:.4f}'.ljust(print_num_minus, ' '), '|')
            with torch.no_grad():
                model.eval()
                gating_model.eval()
                for i, data in enumerate(dataloaders['val']):

                    inputs_val1 = data['image'].float().cuda()
                    mask_val = data['mask'].cuda()
                    sdf_val = data['SDF'].cuda()
                    boundary_val = data['boundary'].cuda()
                    name_val = data['ID']

                    outputs = model(inputs_val1)
                    feat1 = outputs["seg"][0]
                    outputs_val1 = outputs["seg"][1]
                    feat2 = outputs["sdf"][0]
                    outputs_val2 = outputs["sdf"][1]
                    feat3 = outputs["bnd"][0]
                    outputs_val3 = outputs["bnd"][1]
                    gating_input = torch.cat([feat1, feat2, feat3], dim=1)
                    val_out1, val_out2, val_out3 = gating_model(gating_input)

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
                    save_models = {
                        'model': model,
                        'gating_model': gating_model
                    }
                    best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, save_models, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'MoE')

                    if logger is not None:
                        row = {
                            "epoch": epoch + 1,
                            "loss_seg": train_epoch_loss_sup1,
                            "loss_sdf": train_epoch_loss_sup2,
                            "loss_bnd": train_epoch_loss_sup3,
                            "loss_unsup": train_epoch_loss_unsup,
                            "loss_total": train_epoch_loss,
                            "jc": train_jc,
                            "dc": train_dc,
                            "unsup_weight": unsup_weight,
                        }
                        if uncert_stats["count"] > 0:
                            c = uncert_stats["count"]

                            row.update({
                                "U_seg": uncert_stats["U_seg"] / c,
                                "U_sdf": uncert_stats["U_sdf"] / c,
                                "U_bnd": uncert_stats["U_bnd"] / c,

                                "W_seg": uncert_stats["W_seg"] / c,
                                "W_sdf": uncert_stats["W_sdf"] / c,
                                "W_bnd": uncert_stats["W_bnd"] / c,

                                "raw_seg": uncert_stats["raw_seg"] / c,
                                "raw_sdf": uncert_stats["raw_sdf"] / c,
                                "raw_bnd": uncert_stats["raw_bnd"] / c,

                                "weighted_seg": uncert_stats["weighted_seg"] / c,
                                "weighted_sdf": uncert_stats["weighted_sdf"] / c,
                                "weighted_bnd": uncert_stats["weighted_bnd"] / c,

                                "U_seg_mean": uncert_seg_dist["mean"] / c,
                                "U_seg_max": uncert_seg_dist["max"] / c,
                                "U_seg_min": uncert_seg_dist["min"] / c,
                            })
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

















