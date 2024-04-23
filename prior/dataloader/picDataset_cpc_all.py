import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transformfunction = transforms.Compose([
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


class PicDataset_cpc_all(Dataset):
    def __init__(self, csv_file, root_dir, transform=transformfunction):
        self.root_dir = root_dir
        self.transform = transform

        # 读取并处理CSV文件
        dataframe = pd.read_csv(csv_file)
        self.unique_images = dataframe.drop_duplicates(subset='name')

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        img_name = self.unique_images.iloc[idx]['name']
        print(img_name)
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


def main():
    # 图片预处理

    # 设置CSV文件和图像根目录的路径
    csv_file = '/home/zhangshuo/pic/datasets/csv_train/A1AHDLWF5AQ8W_train.csv'
    root_dir = '/public/datasets/CPCDataset/images'

    # 创建Dataset和DataLoader
    dataset = PicDataset_cpc_all(
        csv_file=csv_file, root_dir=root_dir, transform=transformfunction)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 遍历几个批次以演示
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(batch.shape)


if __name__ == "__main__":
    main()
