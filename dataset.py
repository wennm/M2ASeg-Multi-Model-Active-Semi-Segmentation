import numpy as np
import torch
import os

from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# *获取训练、测试集，根据数据集不同分别处理
def get_dataset(name, data_path,P_masks_path, model_name, seed=None):

    if name == 'hyper_kvasir_seg':
        return get_hyper_kvasir_seg_new(data_path, P_masks_path,model_name,train_ratio=0.8, val_ratio=0.1, seed=None)
    elif name in ['ISIC2017','ISIC2018']:
        return get_ISIC2017(data_path, P_masks_path, model_name)









def get_hyper_kvasir_seg_new(data_path, P_masks_path, model_name, train_ratio=0.8, val_ratio=0.1, seed=None):

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Collect image and mask file paths
    image_files = []
    mask_files = []
    p_mask_files = []

    images_dir = os.path.join(data_path, 'images')
    masks_dir = os.path.join(data_path, 'masks')
    p_masks_dir = P_masks_path

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_files.append(os.path.join(images_dir, filename))
            mask_files.append(os.path.join(masks_dir, filename))  # Assuming real masks have the same filename
            p_mask_files.append(os.path.join(p_masks_dir, filename))  # Assuming pseudo masks have the same filename

    # Total dataset size
    dataset_size = len(image_files)
    # Calculate sizes for train, val, and test sets
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Shuffle dataset indices
    indices = torch.randperm(dataset_size)

    # Split indices into train, val, and test
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Define data transformations

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Load train dataset
    X_tr = torch.stack([transform_train(Image.open(image_files[i])) for i in train_indices])
    Y_tr = torch.stack([transform_test(Image.open(mask_files[i])) for i in train_indices])
    Y_Ptr = torch.stack([transform_test(Image.open(p_mask_files[i])) for i in train_indices])

    # Load validation dataset
    X_val = torch.stack([transform_train(Image.open(image_files[i])) for i in val_indices])
    Y_val = torch.stack([transform_test(Image.open(mask_files[i])) for i in val_indices])

    # Load test dataset
    X_te = torch.stack([transform_train(Image.open(image_files[i])) for i in test_indices])
    Y_te = torch.stack([transform_test(Image.open(mask_files[i])) for i in test_indices])

    # Save dataset splits to JSON files
    train_data = {
        "images": [image_files[i] for i in train_indices.tolist()],
        "masks": [mask_files[i] for i in train_indices.tolist()],
        "p_masks": [p_mask_files[i] for i in train_indices.tolist()]
    }
    val_data = {
        "images": [image_files[i] for i in val_indices.tolist()],
        "masks": [mask_files[i] for i in val_indices.tolist()]
    }
    test_data = {
        "images": [image_files[i] for i in test_indices.tolist()],
        "masks": [mask_files[i] for i in test_indices.tolist()]
    }

    with open('train.json', 'w') as train_file:
        json.dump(train_data, train_file, indent=4)
    with open('val.json', 'w') as val_file:
        json.dump(val_data, val_file, indent=4)
    with open('test.json', 'w') as test_file:
        json.dump(test_data, test_file, indent=4)

    return X_tr, Y_tr, Y_Ptr, X_te, Y_te, X_val, Y_val


def get_ISIC2017(data_path, P_masks_path, model_name):
    # 初始化文件路径列表
    train_image_files, train_mask_files, train_p_mask_files = [], [], []
    val_image_files, val_mask_files, val_p_mask_files = [], [], []
    test_image_files, test_mask_files, test_p_mask_files = [], [], []

    # 定义数据转换

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # 处理 train 数据集
    train_images_dir = os.path.join(data_path, 'train', 'images')
    train_masks_dir = os.path.join(data_path, 'train', 'masks')
    train_p_masks_dir = os.path.join(P_masks_path)

    for filename in os.listdir(train_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            train_image_files.append(os.path.join(train_images_dir, filename))
            train_mask_files.append(os.path.join(train_masks_dir, filename))
            train_p_mask_files.append(os.path.join(train_p_masks_dir, filename))

    # 处理 val 数据集
    val_images_dir = os.path.join(data_path, 'val', 'images')
    val_masks_dir = os.path.join(data_path, 'val', 'masks')
    val_p_masks_dir = os.path.join(P_masks_path, 'val')

    for filename in os.listdir(val_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            val_image_files.append(os.path.join(val_images_dir, filename))
            val_mask_files.append(os.path.join(val_masks_dir, filename))
            val_p_mask_files.append(os.path.join(val_p_masks_dir, filename))

    # 处理 test 数据集
    test_images_dir = os.path.join(data_path, 'test', 'images')
    test_masks_dir = os.path.join(data_path, 'test', 'masks')
    test_p_masks_dir = os.path.join(P_masks_path, 'test')

    for filename in os.listdir(test_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            test_image_files.append(os.path.join(test_images_dir, filename))
            test_mask_files.append(os.path.join(test_masks_dir, filename))
            test_p_mask_files.append(os.path.join(test_p_masks_dir, filename))

    # 加载数据并应用转换
    X_tr = torch.stack([transform_train(Image.open(img_path)) for img_path in train_image_files])
    Y_tr = torch.stack([transform_test(Image.open(mask_path)) for mask_path in train_mask_files])
    Y_Ptr = torch.stack([transform_test(Image.open(p_mask_path)) for p_mask_path in train_p_mask_files])

    X_val = torch.stack([transform_train(Image.open(img_path)) for img_path in val_image_files])
    Y_val = torch.stack([transform_test(Image.open(mask_path)) for mask_path in val_mask_files])

    X_te = torch.stack([transform_train(Image.open(img_path)) for img_path in test_image_files])
    Y_te = torch.stack([transform_test(Image.open(mask_path)) for mask_path in test_mask_files])

    return X_tr, Y_tr, Y_Ptr, X_te, Y_te, X_val, Y_val



# *重载data类，编写合适的dataloader
def get_handler(name,model_name):

    if name == 'SSL_Dataset':
        return SSLDataHandler
    elif name in ['ISIC2017','ISIC2018','hyper_kvasir_seg']:
            return SegmentationDataset

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, original_indices, transform=None, transform_params=None):
        self.images = images
        self.masks = masks
        self.original_indices = original_indices  # 新增：存储原始全局索引
        self.transform = transform
        self.transform_params = transform_params or {}

        self.transform_train_mask = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=Image.NEAREST),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
        ])
        self.transform_test_mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
        ])


        # 定义基础数据增强
        self.augmentation = A.Compose([

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(p=0.5),
        ])
        self.augmentation_B = A.Compose([
            A.Resize(224, 224, interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(p=0.5),
        ])

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        # 将张量转换为 PIL 图像
        image = transforms.ToPILImage()(image).convert('RGB')
        mask = transforms.ToPILImage()(mask).convert('L')

        if len(self.transform.transforms) == 2:
            # 将 PIL 图像转换为 NumPy 数组
            image = np.array(image)
            mask = np.array(mask)
            # 应用数据增强
            augmented_A = self.augmentation(image=image, mask=mask)
            # augmented_B = self.augmentation_B(image=image, mask=mask)
            image = augmented_A['image']
            mask = augmented_A['mask']

            image = self.transform(image)
            mask = self.transform_train_mask(mask)
        elif len(self.transform.transforms) == 3:

            image = self.transform(image)
            mask = self.transform_test_mask(mask)

        return image, mask, self.original_indices[index]

    def __len__(self):
        return len(self.images)

class SSLDataHandler:
    def __init__(self, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, transform=None):
        """
        :param transform: 统一的数据增强操作
        """
        self.X_labeled = X_labeled
        self.Y_labeled = Y_labeled
        self.X_unlabeled = X_unlabeled
        self.Y_unlabeled = Y_unlabeled
        self.transform = transform  # 保存transform参数

    def __getitem__(self, index):
        # 应用统一的transform
        x_labeled = self.transform(self.X_labeled[index]) if self.transform else self.X_labeled[index]
        y_labeled = self.Y_labeled[index]
        x_unlabeled = self.transform(self.X_unlabeled[index]) if self.transform else self.X_unlabeled[index]
        y_unlabeled = self.Y_unlabeled[index]
        return (x_labeled, y_labeled), (x_unlabeled, y_unlabeled)

    def __len__(self):
        return len(self.X_labeled)



