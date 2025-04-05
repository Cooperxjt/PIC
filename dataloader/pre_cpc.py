import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CPC(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = []
        self.labels = []
        self.classes = os.listdir(main_dir)
        self.class_to_idx = {
            cls_name: i for i,
            cls_name in enumerate(self.classes)
        }

        for cls_name in self.classes:
            class_dir = os.path.join(main_dir, cls_name)
            for img_name in os.listdir(class_dir):
                self.total_imgs.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # 图片转换处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])

    # 创建数据集
    dataset = CPC(
        main_dir="/public/datasets/CPCDataset/process/user_best_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 模拟训练循环
    for epoch in range(2):  # 假设训练2个epoch
        for i, (images, labels) in enumerate(dataloader):
            print(
                f"Epoch: {epoch}, Batch: {i}, Image shape: {images.shape}, Labels: {labels}")
            break
        # 在这里添加模型训练代码


if __name__ == "__main__":
    main()
