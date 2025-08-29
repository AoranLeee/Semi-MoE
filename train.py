import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.getnetwork import get_network
import argparse
import time
import os
import numpy as np
import random
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

simplefilter(action='ignore', category=FutureWarning)

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "16672"

def create_model(network, in_channels, num_classes, **kwargs):
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='CRAG', help='CREMI, GlaS, ISIC-2017')
    parser.add_argument('--sup_mark', default='35')
    parser.add_argument('--unsup_mark', default='138')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=0.5, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='unet', type=str)
    parser.add_argument('-gn', '--gating_network', default='multi_gating_attention', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('--visdom_port', default=16672)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # trained model save
    path_trained_models = cfg['PATH_TRAINED_MODEL'] + '/' + str(dataset_name)
    if rank == args.rank_index:
        os.makedirs(path_trained_models, exist_ok=True)
    path_trained_models = path_trained_models + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    if rank == args.rank_index:
        os.makedirs(path_trained_models, exist_ok=True)

    # seg results save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if rank == args.rank_index:
        os.makedirs(path_seg_results, exist_ok=True)
    path_seg_results = path_seg_results + '/' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-uw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' + str(args.sup_mark) + '-' + str(args.unsup_mark)
    if rank == args.rank_index:
        os.makedirs(path_seg_results, exist_ok=True)

    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_train_unsup = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/train_unsup_' + args.unsup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=False,
        num_images=None,
    )
    num_images_unsup = len(dataset_train_unsup)

    dataset_train_sup = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/train_sup_' + args.sup_mark,
        data_transform_1=data_transforms['train'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None,
    )

    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup, shuffle=True)
    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    segment_model = create_model(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    sdf_model = create_model(args.network, cfg['IN_CHANNELS'], 1)
    boundary_model = create_model(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    gating_model = create_model(args.gating_network, cfg['IN_CHANNELS'] * 64, cfg['NUM_CLASSES'])

    dist.barrier()

    criterion = segmentation_loss(args.loss, False).cuda()
    loss_fn = MultiTaskLoss().cuda()

    optimizer1 = optim.SGD(segment_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    optimizer2 = optim.SGD(sdf_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup2 = GradualWarmupScheduler(optimizer2, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler2)


    optimizer3 = optim.SGD(boundary_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup3 = GradualWarmupScheduler(optimizer3, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler3)

    optimizer4 = optim.SGD(gating_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler4 = lr_scheduler.StepLR(optimizer4, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup4 = GradualWarmupScheduler(optimizer4, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler4)

    optimizer5 = optim.Adam(loss_fn.parameters(), 0.05, weight_decay=5 * 10 ** args.wd)


    since = time.time()
    count_iter = 0

    best_model = segment_model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        segment_model.train()
        sdf_model.train()
        boundary_model.train()
        gating_model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_sup_3 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0
        val_loss_sup_3 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        dist.barrier()

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup1 = unsup_index['image'].float().cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
        
            feat_unsup1, pred_train_unsup1 = segment_model(img_train_unsup1)
            feat_unsup2, pred_train_unsup2 = sdf_model(img_train_unsup1)
            feat_unsup3, pred_train_unsup3 = boundary_model(img_train_unsup1)

            gating_unsup_input = torch.cat([feat_unsup1, feat_unsup2, feat_unsup3], dim=1)
            unsup_out1, unsup_out2, unsup_out3 = gating_model(gating_unsup_input)

            fake_bnd = torch.max(unsup_out3, dim=1)[1].detach()
            fake_sdf = (torch.tanh(unsup_out2)).detach()
            fake_mask = torch.max(unsup_out1, dim=1)[1].long().detach()
            
            loss_unsup_seg = criterion(pred_train_unsup1, fake_mask)
            loss_unsup_sdf = sdf_loss(torch.tanh(pred_train_unsup2), fake_sdf)
            loss_unsup_bnd = imbalance_diceLoss(pred_train_unsup3, fake_bnd)
            loss_train_unsup = loss_fn(loss_unsup_seg, loss_unsup_sdf, loss_unsup_bnd)

            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup1 = sup_index['image'].float().cuda()
            mask_train_sup = sup_index['mask'].cuda()
            sdf_train_sup = sup_index['SDF'].cuda()
            boundary_train_sup = sup_index['boundary'].cuda()
 
            feat_sup1, pred_train_sup1 = segment_model(img_train_sup1)
            feat_sup2, pred_train_sup2 = sdf_model(img_train_sup1)
            feat_sup3, pred_train_sup3 = boundary_model(img_train_sup1)

            gating_sup_input = torch.cat([feat_sup1, feat_sup2, feat_sup3], dim=1)
            sup_out1, sup_out2, sup_out3 = gating_model(gating_sup_input)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = sup_out1
                    mask_list_train = mask_train_sup
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train1 = torch.cat((score_list_train1, sup_out1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = (criterion(pred_train_sup1, mask_train_sup) + criterion(sup_out1, mask_train_sup))
            loss_train_sup2 = sdf_loss(torch.tanh(pred_train_sup2), sdf_train_sup) + sdf_loss(torch.tanh(sup_out2), sdf_train_sup) 
            loss_train_sup3 = imbalance_diceLoss(pred_train_sup3, boundary_train_sup) + imbalance_diceLoss(sup_out3, boundary_train_sup)

            loss_train_sup = loss_fn(loss_train_sup1, loss_train_sup2, loss_train_sup3)
   
            loss_train_sup.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        scheduler_warmup2.step()
        scheduler_warmup3.step()
        scheduler_warmup4.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            score_gather_list_train1 = [torch.zeros_like(score_list_train1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train1, score_list_train1)
            score_list_train1 = torch.cat(score_gather_list_train1, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_sup3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
                train_eval_list1, train_m_jc1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
                torch.cuda.empty_cache()

            with torch.no_grad():
                segment_model.eval()
                sdf_model.eval()
                boundary_model.eval()
                gating_model.eval()
                for i, data in enumerate(dataloaders['val']):

                    inputs_val1 = data['image'].float().cuda()
                    mask_val = data['mask'].cuda()
                    sdf_val = data['SDF'].cuda()
                    boundary_val = data['boundary'].cuda()
                    name_val = data['ID']

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()
                    optimizer4.zero_grad()
                    optimizer5.zero_grad()
               
                    feat1, outputs_val1 = segment_model(inputs_val1)
                    feat2, outputs_val2 = sdf_model(inputs_val1)
                    feat3, outputs_val3 = boundary_model(inputs_val1)
                    gating_input = torch.cat([feat1, feat2, feat3], dim=1)
                    val_out1, val_out2, val_out3 = gating_model(gating_input)

                    torch.cuda.empty_cache()

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

                torch.cuda.empty_cache()
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
                        'segment_model': segment_model,
                        'sdf_model': sdf_model,
                        'boundary_model': boundary_model,
                        'gating_model': gating_model
                    }
                    best_val_eval_list = save_val_best_sup_2d(cfg['NUM_CLASSES'], best_val_eval_list, save_models, score_list_val1, name_list_val, val_eval_list1, path_trained_models, path_seg_results, cfg['PALETTE'], 'MoE')
                    torch.cuda.empty_cache()

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(
                        print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
        print('=' * print_num)