import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pywt
import random
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from skimage.morphology import dilation, disk


def compute_sdf(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    posmask = np.array(img_gt, dtype=bool)
    negmask = ~posmask
    posdis = distance(posmask)
    negdis = distance(negmask)
    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)

    posdis = (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
    negdis = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis))

    sdf = posdis - negdis 

    sdf[boundary == 1] = 0
    assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
    assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return torch.tensor(sdf, dtype=torch.float32)

def get_gt_bnd(gt):
    gt = (np.array(gt) > 0).astype(np.uint8)  
    bnd = np.zeros_like(gt, dtype=np.uint8)
    
    for j in range(1, gt.max() + 1):
        _gt = (gt == j).astype(np.uint8) 
        _gt_dil = dilation(_gt, disk(2)) 
        bnd[_gt_dil - _gt == 1] = 1  
    return torch.tensor(bnd, dtype=torch.float32)

class DatasetSemiMoE(Dataset):
    def __init__(self, data_dir, augmentation_1, normalize_1, sup=True, num_images=None, **kwargs):
        super(DatasetSemiMoE, self).__init__()

        img_paths_1 = []
        mask_paths = []

        image_dir_1 = data_dir + '/image'
        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_1):

            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img_1 = Image.open(img_path_1)
        img_1 = np.array(img_1)
      

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment_1 = self.augmentation_1(image=img_1, mask=mask)
            img_1 = augment_1['image']
            mask_1 = augment_1['mask']
            
            normalize_1 = self.normalize_1(image=img_1, mask=mask_1)
            img_1 = normalize_1['image']
            mask_1 = normalize_1['mask'].long()
            sdf = compute_sdf(mask_1)
            bnd = get_gt_bnd(mask_1)

            sampel = {'image': img_1, 'mask': mask_1, 'SDF': sdf, 'boundary': bnd, 'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img_1)
            img_1 = augment_1['image']

            normalize_1 = self.normalize_1(image=img_1)
            img_1 = normalize_1['image']

            sampel = {'image': img_1, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def get_imagefolder(data_dir, data_transform_1, data_normalize_1, sup=True, num_images=None, **kwargs):
    dataset = DatasetSemiMoE(data_dir=data_dir,
                          augmentation_1=data_transform_1,
                          normalize_1=data_normalize_1,
                          sup=sup,
                          num_images=num_images,
                           **kwargs)
    return dataset
