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
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomCrop(256, padding=8, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]
    )
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

        annotations_data = []
        for _, row in annotations.iterrows():
            bbox = [int(x) for x in row['bbox'].strip('[]').split(',')]
            score = row['score']
            annotations_data.append({'bbox': bbox, 'score': score})
        sample = {'image': image, 'annotations': annotations_data}

        if self.transform:
            sample = self.transform(sample, self.image_size)

        cpc_image = Image.open(img_name)
        cpc_image = self.cpc_tranform(cpc_image)

        sample['cpc_image'] = cpc_image
        sample['img_name'] = img_name

        return sample
