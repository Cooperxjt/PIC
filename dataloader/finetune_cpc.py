import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

cpc_transformfunction = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小至256x256
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转（-20到20度）
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),  # 随机调整亮度、对比度、饱和度和色调
    transforms.RandomCrop(256, padding=8, pad_if_needed=True),  # 随机裁剪
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]
    )  # 标准化
])


class TransformFunction(object):

    def __call__(self, sample, image_size):
        image, annotations = sample['image'], sample['annotations']

        scale = float(image_size) / float(min(image.shape[:2]))
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = float(resized_image.shape[0]) / image.shape[0]
        scale_width = float(resized_image.shape[1]) / image.shape[1]

        transformed_bbox = {}
        transformed_bbox['xmin'] = []
        transformed_bbox['ymin'] = []
        transformed_bbox['xmax'] = []
        transformed_bbox['ymax'] = []
        MOS = []
        for annotation in annotations:
            transformed_bbox['xmin'].append(
                math.floor(float(annotation['bbox'][0]) * scale_width))
            transformed_bbox['ymin'].append(
                math.floor(float(annotation['bbox'][1]) * scale_height))
            transformed_bbox['xmax'].append(
                math.ceil(float(annotation['bbox'][2]) * scale_width))
            transformed_bbox['ymax'].append(
                math.ceil(float(annotation['bbox'][3]) * scale_height))
            MOS.append(annotation['score'])

        resized_image = resized_image.transpose((2, 0, 1))

        # print(resized_image.shape)
        return {'image': resized_image, 'bbox': transformed_bbox, 'MOS': MOS, 'scale_width': scale_width, 'scale_height': scale_height}


class CPCDataset_pic(Dataset):
    def __init__(
            self, csv_file, image_size=256, dataset_dir='dataset/cpc/', set='train',
            transform=TransformFunction(), augmentation=False
    ):
        """
        csv_file (string): 包含注释的csv文件的路径。
        root_dir (string): 包含所有图像的目录。
        transform (callable, optional): 可选的变换，应用于图像。
        """
        self.annotations_frame = pd.read_csv(csv_file)
        self.image_annotations = self._group_annotations()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_size = image_size
        self.cpc_tranform = cpc_transformfunction

    def _group_annotations(self):
        grouped = self.annotations_frame.groupby('name')
        return {name: group[['bbox', 'score']] for name, group in grouped}

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, list(
            self.image_annotations.keys())[idx])

        image = cv2.imread(img_name)
        annotations = self.image_annotations[list(
            self.image_annotations.keys())[idx]]

        # 解析 bbox 和 score
        annotations_data = []
        for _, row in annotations.iterrows():
            bbox = [int(x) for x in row['bbox'].strip('[]').split(',')]
            score = row['score']
            annotations_data.append({'bbox': bbox, 'score': score})
        sample = {'image': image, 'annotations': annotations_data}

        if self.transform:
            sample = self.transform(sample, self.image_size)

        # 处理 个人特征 使用的那部分图片
        cpc_image = Image.open(img_name)
        cpc_image = self.cpc_tranform(cpc_image)

        sample['cpc_image'] = cpc_image
        sample['img_name'] = img_name

        return sample


# 使用 CustomDataset 创建 DataLoader


# if __name__ == '__main__':

#     # 请根据您的文件路径修改以下两个参数
#     csv_file = '/home/zhangshuo/pic/datasets/finetune/3/val/A1DINOR5IK1YC4_val.csv'
#     root_dir = '/public/datasets/CPCDataset/images'

#     dataset = CPCDataset_pic(
#         csv_file=csv_file, dataset_dir=root_dir, image_size=256
#     )
#     data_loader = DataLoader(
#         dataset, batch_size=1,
#         shuffle=True, num_workers=0
#     )

#     for i, batch in enumerate(data_loader):
#         print(f'Batch {i+1}')
#         print(f'Images in this batch: {len(batch["image"])}')
#         print('Annotations for the first image:')
#         # print(batch['bbox'])
#         # print(batch['MOS'])
#         # 如果您想看到每个图像的注释，可以取消注释以下行
#         # for annotation in batch['annotations']:
#         #     print(annotation)
#         # 限制输出以避免太长的日志
#         if i == 1:  # 只迭代两个批次作为示例
#             break
