import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.utils.data as data
import cv2
import math
import numpy as np
from augmentations import CropAugmentation
import random


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class TransformFunction(object):

    def __call__(self, sample, image_size):
        image, ori_boxes = sample['image'], sample['ori_bboxes']

        # 将图片resize为image_size*image_size
        resized_image = cv2.resize(
            image, (int(image_size), int(image_size))) / 256.0
        # scale = float(image_size) / float(min(image.shape[:2]))

        # h = round(image.shape[0] * scale / 32.0) * 32
        # w = round(image.shape[1] * scale / 32.0) * 32

        # resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)

        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = float(image_size) / image.shape[0]
        scale_width = float(image_size) / image.shape[1]

        transformed_bbox = {}
        transformed_bbox['xmin'] = []
        transformed_bbox['ymin'] = []
        transformed_bbox['xmax'] = []
        transformed_bbox['ymax'] = []

        for box in ori_boxes:
            transformed_bbox['xmin'].append(
                math.floor(float(box[1]) * scale_width))
            transformed_bbox['ymin'].append(
                math.floor(float(box[0]) * scale_height))
            transformed_bbox['xmax'].append(
                math.ceil(float(box[3]) * scale_width))
            transformed_bbox['ymax'].append(
                math.ceil(float(box[2]) * scale_height))

        resized_image = resized_image.transpose((2, 0, 1))

        sample['image'] = resized_image
        sample['transformed_bboxes'] = transformed_bbox

        return sample


class CPC(Dataset):
    def __init__(self, image_size, csv_folder, txt_folder, img_folder, set='train',
                 augmentation=False, transform=TransformFunction()):
        self.set = set
        self.img_folder = img_folder
        self.image_size = image_size
        self.transform = transform

        if augmentation:
            self.augmentation = CropAugmentation()
        else:
            self.augmentation = None

        self.user_data = {}

        for csv_file in os.listdir(csv_folder):
            user = os.path.splitext(csv_file)[0]

            if user not in self.user_data:
                self.user_data[user] = []

            df = pd.read_csv(os.path.join(csv_folder, csv_file))

            grouped = df.groupby('name')

            for image_name, group in grouped:
                txt_file_path = os.path.join(
                    txt_folder, 'processed_' + image_name + '.txt')

                if os.path.exists(txt_file_path):
                    # 有可能对应的图片并不存在
                    user_scores = group['score'].tolist()

                    # 添加图片对应的平均分数
                    with open(txt_file_path) as f:
                        avg_scores = [
                            float(line.split()[-1])
                            for line in f.readlines()
                        ]

                    # 添加裁剪框，这个box的顺序为xmin ymin xmax ymax
                    with open(txt_file_path) as f:
                        boxes = [
                            [float(num) for num in line.split()[:-1]]
                            for line in f.readlines()
                        ]

                    # 将这个用户的信息添加进来
                    self.user_data[user].append(
                        (image_name, boxes, user_scores, avg_scores)
                    )

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        # user, image_name, boxes, user_scores, avg_scores = self.data[idx]
        user = list(self.user_data.keys())[idx]
        user_info = self.user_data[user]

        images_info = {}

        # 如果用户的图片数量大于 200，则随机选择150张图片
        if len(user_info) > 64:
            user_info = random.sample(user_info, 64)

        for image_info in user_info:
            # 处理该用户的多张图片
            image_name, boxes, user_scores, avg_scores = image_info

            image_path = os.path.join(self.img_folder, image_name)
            image = cv2.imread(image_path)

            score_diffs = [user_score - avg_score for user_score,
                           avg_score in zip(user_scores, avg_scores)]

            # 将boxes的顺序调整为和 gaic的一致，ymin xmin ymax xmax
            for box in boxes:
                box[0], box[1] = box[1], box[0]
                box[2], box[3] = box[3], box[2]

            if self.augmentation:
                image, boxes = self.augmentation(image, boxes)

            # to rgb
            image = image[:, :, (2, 1, 0)]

            images_info[image_name] = {
                'image': image,
                'ori_bboxes': boxes,
                'user_scores': user_scores,
                'avg_scores': avg_scores,
                'score_diffs': score_diffs
            }

            if self.transform:
                images_info[image_name] = self.transform(
                    images_info[image_name], self.image_size
                )

        sample = {
            'user': user,
            'images': images_info
        }

        return sample


def custom_collate_fn(batch):
    # 这里的实现需要根据您的具体数据结构进行调整
    collated_batch = {'user': [], 'images': []}
    for item in batch:
        collated_batch['user'].append(item['user'])
        collated_batch['images'].append(item['images'])
    return collated_batch


def main():
    csv_folder = '/public/datasets/CPCDataset/process/user_score_process'
    txt_folder = '/public/datasets/CPCDataset/process/avg_results'
    img_folder = '/public/datasets/CPCDataset/images'

    dataset = CPC(256, csv_folder, txt_folder, img_folder)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=custom_collate_fn)

    for data in dataloader:
        user = data['user']
        print(f"User: {user}")
        images_info_list = data['images']
        for images_info in images_info_list:
            for image_name, image_data in images_info.items():
                print(f"Image Name: {image_name}")
                print("Original Boxes:", image_data['ori_bboxes'])
                print("Transformed Boxes:", image_data['transformed_bboxes'])
                print("User Scores:", image_data['user_scores'])
                print("Average Scores:", image_data['avg_scores'])
                print("Score Differences:", image_data['score_diffs'])
        break


if __name__ == "__main__":
    main()
