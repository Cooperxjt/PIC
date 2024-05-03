import os
import pickle

import numpy as np
from PIL import Image


def get_pdefined_anchors(anchor_file):
    with open(anchor_file, 'rb') as file:
        anchors = pickle.load(file, encoding='bytes')
    return np.array(anchors)


img_name = '6808.jpg'  # 请确保这里使用正确的图像路径
image = Image.open(img_name)

img_width, img_height = image.size

np.set_printoptions(threshold=np.inf)
anchors = get_pdefined_anchors('pdefined_anchor.pkl')

# 创建一个目录来保存裁剪的图像
output_dir = 'cropped_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(range(len(anchors)))

for i, anchor in enumerate(anchors):
    bbox = [float(x) for x in anchor[:4]]
    print(bbox)
    x1 = int(bbox[0] * img_width)
    y1 = int(bbox[1] * img_height)
    x2 = int(bbox[2] * img_width)
    y2 = int(bbox[3] * img_height)

    # 裁剪图像
    cropped_img = image.crop((x1, y1, x2, y2))

    # 保存裁剪的图像
    cropped_img.save(f"{output_dir}/cropped_image_{i}.jpg")
    print(f"Saved cropped_image_{i}.jpg")
