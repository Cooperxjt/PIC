import math
import os
import pickle
import random

import cv2
import numpy as np
import pandas as pd
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

        ori_bbox = {}
        ori_bbox['xmin'] = []
        ori_bbox['ymin'] = []
        ori_bbox['xmax'] = []
        ori_bbox['ymax'] = []

        MOS = []
        for annotation in annotations:
            transformed_bbox['xmin'].append(
                math.floor(float(annotation['bbox'][0]) * scale_width)
            )
            transformed_bbox['ymin'].append(
                math.floor(float(annotation['bbox'][1]) * scale_height)
            )
            transformed_bbox['xmax'].append(
                math.ceil(float(annotation['bbox'][2]) * scale_width)
            )
            transformed_bbox['ymax'].append(
                math.ceil(float(annotation['bbox'][3]) * scale_height)
            )

            ori_bbox['xmin'].append(
                math.floor(float(annotation['bbox'][0]))
            )
            ori_bbox['ymin'].append(
                math.floor(float(annotation['bbox'][1]))
            )
            ori_bbox['xmax'].append(
                math.ceil(float(annotation['bbox'][2]))
            )
            ori_bbox['ymax'].append(
                math.ceil(float(annotation['bbox'][3]))
            )
            MOS.append(annotation['score'])

            # if annotation['score'] == 4:
            #     print(annotation['bbox'][0] * scale_width)
            #     print(annotation['bbox'][1] * scale_height)
            #     print(annotation['bbox'][2] * scale_width)
            #     print(annotation['bbox'][3] * scale_height)

        resized_image = resized_image.transpose((2, 0, 1))

        # print(resized_image.shape)
        return {'image': resized_image, 'bbox': transformed_bbox, 'MOS': MOS, 'ori_bbox': ori_bbox}


class CUHKDataset_pic(Dataset):
    def __init__(
            self, csv_file, image_size=256, dataset_dir='All_Images',
            transform=TransformFunction(), augmentation=True, set='Train'
    ):
        """
        csv_file (string): 包含注释的csv文件的路径。
        root_dir (string): 包含所有图像的目录。
        transform (callable, optional): 可选的变换，应用于图像。
        """
        self.annotations_frame = pd.read_csv(csv_file)

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_size = image_size
        self.cpc_tranform = cpc_transformfunction
        self.set = set
        self.anchors = self.get_pdefined_anchors(
            'dataloader/pdefined_anchor.pkl')
        if self.set == 'train':
            np.random.shuffle(self.anchors)

    def get_pdefined_anchors(self, anchor_file):
        anchors = pickle.load(open(anchor_file, 'rb'), encoding='bytes')
        anchors = np.array(anchors)

        return anchors

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.dataset_dir,
            list(self.annotations_frame['Image'])[idx].split('\\')[1]
        )
        image = cv2.imread(img_name)

        crop_box = list(self.annotations_frame['Crop Box'])[idx]

        # 解析 原始的bbox 和 score，将其置为最高分。
        annotations_data = []

        bbox = [int(x) for x in crop_box.split()]
        score = 4
        annotations_data.append({'bbox': bbox, 'score': score})

        img_height = image.shape[0]
        img_width = image.shape[1]

        # 当为训练模式，添加一些新的裁剪框，并置为零分。
        if self.set == 'train':

            for i in range(23):
                bbox = [float(x) for x in self.anchors[i][:4]]

                x1 = bbox[0] * img_width
                y1 = bbox[1] * img_height
                x2 = bbox[2] * img_width
                y2 = bbox[3] * img_height

                bbox[0] = x1
                bbox[1] = y1
                bbox[2] = x2
                bbox[3] = y2

                score = 0

                annotations_data.append({'bbox': bbox, 'score': score})

        elif self.set == 'val':
            for i in range(len(self.anchors)):
                bbox = [float(x) for x in self.anchors[i][:4]]

                x1 = bbox[0] * img_width
                y1 = bbox[1] * img_height
                x2 = bbox[2] * img_width
                y2 = bbox[3] * img_height

                bbox[0] = x1
                bbox[1] = y1
                bbox[2] = x2
                bbox[3] = y2

                score = 0

                annotations_data.append({'bbox': bbox, 'score': score})

        sample = {'image': image, 'annotations': annotations_data}

        if self.transform:
            sample = self.transform(sample, self.image_size)

        # 处理 个人特征 使用的那部分图片
        cpc_image = Image.open(img_name)
        cpc_image = self.cpc_tranform(cpc_image)

        sample['cpc_image'] = cpc_image
        sample['img_height'] = img_height
        sample['img_width'] = img_width

        return sample


# 使用 CustomDataset 创建 DataLoader


if __name__ == '__main__':

    # 请根据您的文件路径修改以下两个参数
    csv_file = '/public/datasets/cuhk_cropping/expert_1_csv/expert_1_crops.csv'
    dataset_dir = '/public/datasets/cuhk_cropping/All_Images'

    dataset = CUHKDataset_pic(
        csv_file=csv_file, dataset_dir=dataset_dir, image_size=256, set='val'
    )

    data_loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False,
        num_workers=0
    )

    for i, batch in enumerate(data_loader):
        print(f'Batch {i+1}')
        print(f'Images in this batch: {len(batch["image"])}')
        print('Annotations for the first image:')
        print(batch['bbox'])
        print(batch['MOS'])
        if i == 0:  # 只迭代两个批次作为示例
            break
