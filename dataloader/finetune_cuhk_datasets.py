import math
import os
import pickle
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
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


def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    """'
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    """

    gt_crop = gt_crop[gt_crop[:, 0] >= 0]

    zero_t = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:, 0], pre_crop[:, 0])
    over_y1 = torch.maximum(gt_crop[:, 1], pre_crop[:, 1])
    over_x2 = torch.minimum(gt_crop[:, 2], pre_crop[:, 2])
    over_y2 = torch.minimum(gt_crop[:, 3], pre_crop[:, 3])
    over_w = torch.maximum(zero_t, over_x2 - over_x1)
    over_h = torch.maximum(zero_t, over_y2 - over_y1)

    inter = over_w * over_h

    area1 = (gt_crop[:, 2] - gt_crop[:, 0]) * (gt_crop[:, 3] - gt_crop[:, 1])
    area2 = (pre_crop[:, 2] - pre_crop[:, 0]) * \
        (pre_crop[:, 3] - pre_crop[:, 1])

    union = area1 + area2 - inter
    iou = inter / union

    disp = ((
        torch.abs(gt_crop[:, 0] - pre_crop[:, 0])
        + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])
    ) / im_w + (
        torch.abs(gt_crop[:, 1] - pre_crop[:, 1])
        + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])
    ) / im_h) / 4

    iou_idx = torch.argmax(iou, dim=-1)

    dis_idx = torch.argmin(disp, dim=-1)

    index = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx

    return iou[index].item(), disp[index].item()


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
        # 决定了生成裁剪框的方式，选择预定义还是生成函数
        self.generate_anchors = True
        # self.anchors = self.get_pdefined_anchors('dataloader/pdefined_anchor_6.pkl')

    def get_pdefined_anchors(self, anchor_file):
        anchors = pickle.load(open(anchor_file, 'rb'), encoding='bytes')
        anchors = np.array(anchors)

        return anchors

    def generate_crops(self, width, height):
        # 来自vfn的生成裁剪框方式
        crops = []

        for scale in range(5, 10):
            scale /= 10
            w, h = width * scale, height * scale
            dw, dh = width - w, height - h
            dw, dh = dw / 10, dh / 10

            for w_idx in range(5):
                for h_idx in range(5):
                    # x, y = w_idx * dw, h_idx * dh
                    # crops.append([int(x), int(y), int(w), int(h)])
                    x1, y1 = w_idx * dw, h_idx * dh
                    x2, y2 = x1 + w, y1 + h  # 计算右下角坐标
                    crops.append([int(x1), int(y1), int(x2), int(y2)])

        return crops

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

        gt_crop = [int(x) for x in crop_box.split()]
        score = 10

        annotations_data.append({'bbox': gt_crop, 'score': score})

        img_height = image.shape[0]
        img_width = image.shape[1]

        # 当为训练模式，添加一些新的裁剪框，并置为零分。
        if self.generate_anchors:

            bboxes = self.generate_crops(img_width, img_height)

            if self.set == 'train':
                for bbox in bboxes:

                    pre_bbox = torch.tensor([[x for x in bbox]])
                    gt_bbox = torch.tensor([[x for x in gt_crop]])
                    iou, disp = compute_iou_and_disp(
                        gt_bbox, pre_bbox, img_width, img_height)

                    if iou > 0.9:
                        score = 3
                    elif iou > 0.8 and iou <= 0.9:
                        score = 2
                    else:
                        score = 0

                    annotations_data.append({'bbox': bbox, 'score': score})

                random.shuffle(annotations_data)

            elif self.set == 'val':

                for bbox in bboxes:

                    pre_bbox = torch.tensor([[x for x in bbox]])
                    gt_bbox = torch.tensor([[x for x in gt_crop]])

                    score = 0

                    annotations_data.append({'bbox': bbox, 'score': score})

        else:
            if self.set == 'train':
                np.random.shuffle(self.anchors)

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

                if self.set == 'train':
                    pre_bbox = torch.tensor([[x for x in bbox]])
                    gt_bbox = torch.tensor([[x for x in gt_crop]])
                    iou, disp = compute_iou_and_disp(
                        gt_bbox, pre_bbox, img_width, img_height)

                    if iou > 0.8 and disp < 0.1:
                        score = 3

                    elif iou > 0.8 and disp < 0.15:
                        score = 2
                    else:
                        score = -1
                elif self.set == 'val':
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

    expert_id = 1
    n = 10

    # 对一个用户提取其图片
    val_file = f'/public/datasets/cuhk_cropping/expert_{expert_id}_csv/{n}/expert_{expert_id}_val.csv'
    finetune_file = f'/public/datasets/cuhk_cropping/expert_{expert_id}_csv/{n}/expert_{expert_id}_finetune.csv'

    # 用于返回图片
    dataloader_finetune = data.DataLoader(
        CUHKDataset_pic(
            csv_file=finetune_file,
            image_size=256,
            dataset_dir='/public/datasets/cuhk_cropping/All_Images',
            set='train'
        ),
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )

    for i, batch in enumerate(dataloader_finetune):
        print(i)
        print('\n')
        # print(f'Batch {i+1}')
        # print(f'Images in this batch: {len(batch["image"])}')
        # print('Annotations for the first image:')
        # print(batch['bbox'])
        # print(batch['MOS'])
        # if i == 0:  # 只迭代两个批次作为示例
        #     break
