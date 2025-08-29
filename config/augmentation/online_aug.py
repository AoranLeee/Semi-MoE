import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_transform_2d():
    data_transforms = {
        'train': A.Compose([
            A.Resize(256, 256, p=1),
            A.RandomCrop(height=224, width=224, p=0.5),
            A.Resize(256, 256, p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
        ),
        'val': A.Compose([
            A.Resize(256, 256, p=1),
        ],
        ),
        'test': A.Compose([
            A.Resize(256, 256, p=1),
        ],
        )
    }
    return data_transforms

def data_normalize_2d(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
    )
    return data_normalize


