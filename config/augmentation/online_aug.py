import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_transform_2d():
    data_transforms = {
        'train': A.Compose([
            A.Resize(256, 256, p=1),#缩放/拉伸到 256×256 像素
            A.RandomCrop(height=224, width=224, p=0.5),#有 50% 概率从当前 256×256 图像中随机裁剪出 224×224 的区域
            A.Resize(256, 256, p=1),
            A.Flip(p=0.75),#以 75% 的概率进行水平翻转
            A.Transpose(p=0.5),#以 50% 的概率进行转置（交换行和列）
            A.RandomRotate90(p=1),#始终应用一次按 90° 倍数的随机旋转（旋转角度是 0°, 90°, 180°, 270° 的一种随机选择
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
            A.Normalize(mean, std),#对图像通道进行标准化
            ToTensorV2()#将图像转换为 loat32 的 PyTorch Tensor
        ],
    )
    return data_normalize


