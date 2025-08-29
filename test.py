import torch
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import get_imagefolder
from config.train_test_config.train_test_config import print_test_eval, save_test_2d
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "16672"

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_model(network, in_channels, num_classes, **kwargs):
    model = get_network(network, in_channels, num_classes, **kwargs).cuda()
    return DistributedDataParallel(model, device_ids=[args.local_rank])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_model', default='')
    parser.add_argument('--path_seg_results', default='seg_pred/test')
    parser.add_argument('--dataset_name', default='CRAG', help='GlaS, CRAG')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-n', '--network', default='unet', type=str)
    parser.add_argument('-gn', '--gating_network', default='multi_gating_attention', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    if rank == args.rank_index:
        os.makedirs(os.path.join(args.path_seg_results, str(dataset_name), 
                                os.path.splitext(os.path.basename(args.path_model))[0]), exist_ok=True)

    path_seg_results = os.path.join(args.path_seg_results, str(dataset_name), 
                                    os.path.splitext(os.path.basename(args.path_model))[0])


    # Dataset
    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_test = get_imagefolder(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['test'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)

    dataloaders = dict()
    dataloaders['test'] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=test_sampler)

    num_batches = {'test': len(dataloaders['test'])}

    # Models
    segment_model = create_model(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    sdf_model = create_model(args.network, cfg['IN_CHANNELS'], 1)
    boundary_model = create_model(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    gating_model = create_model(args.gating_network, cfg['IN_CHANNELS'] * 64, cfg['NUM_CLASSES'])

    checkpoint = torch.load(args.path_model, map_location='cuda')
    
    models = {
        'segment_model': segment_model,
        'sdf_model': sdf_model,
        'boundary_model': boundary_model,
        'gating_model': gating_model
    }
    for model_name, model in models.items():
        model.load_state_dict(checkpoint[model_name])
        model.to('cuda')
        model.eval()

    dist.barrier()

    # Test
    since = time.time()

    with torch.no_grad():

        for i, data in enumerate(dataloaders['test']):
            inputs_test = data['image'].float().cuda()
            mask_test = data['mask'].cuda()
            sdf_test = data['SDF'].unsqueeze(1).cuda()
            boundary_test = data['boundary'].cuda()
            name_test = data['ID']

            feat1, outputs_test1 = segment_model(inputs_test)
            feat2, outputs_test2 = sdf_model(inputs_test)
            feat3, outputs_test3 = boundary_model(inputs_test)
            gating_input = torch.cat([feat1, feat2, feat3], dim=1)
            test_out1, test_out2, test_out3 = gating_model(gating_input)

            if i == 0:
                score_list_test = test_out1
                name_list_test = name_test
                mask_list_test = mask_test
            else:
                score_list_test = torch.cat((score_list_test, test_out1), dim=0)
                name_list_test = np.append(name_list_test, name_test, axis=0)
                mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
            torch.cuda.empty_cache()

        score_gather_list_test = [torch.zeros_like(score_list_test) for _ in range(ngpus_per_node)]
        torch.distributed.all_gather(score_gather_list_test, score_list_test)
        score_list_test = torch.cat(score_gather_list_test, dim=0)

        mask_gather_list_test = [torch.zeros_like(mask_list_test) for _ in range(ngpus_per_node)]
        torch.distributed.all_gather(mask_gather_list_test, mask_list_test)
        mask_list_test = torch.cat(mask_gather_list_test, dim=0)

        name_gather_list_test = [None for _ in range(ngpus_per_node)]
        torch.distributed.all_gather_object(name_gather_list_test, name_list_test)
        name_list_test = np.concatenate(name_gather_list_test, axis=0)

        if rank == args.rank_index:
            print('=' * print_num)
            test_eval_list = print_test_eval(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus)
            save_test_2d(cfg['NUM_CLASSES'], score_list_test, name_list_test, test_eval_list[0], path_seg_results, cfg['PALETTE'])
            torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)
