import argparse
import math
import os
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from dataloader.cpcDataset import CPCDataset
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
from tqdm import tqdm

from models.croppingModel import build_crop_model

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0*")


def test():
    net.eval()

    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []

    img_num = len(data_loader_test)

    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for id, sample in enumerate(data_loader_test):

        image = sample['image']
        bboxs = sample['bbox']
        MOS = sample['MOS']

        roi = []

        for idx in range(0, len(bboxs['xmin'])):
            roi.append(
                (
                    0, bboxs['xmin'][idx], bboxs['ymin']
                    [idx], bboxs['xmax'][idx], bboxs['ymax'][idx]
                )
            )

        if cuda:
            image = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            image = Variable(image)
            roi = Variable(roi)

        out = net(image, roi)

        # 打印结果
        id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)

        for k in range(4):
            temp_acc_4_5 = 0.0
            temp_acc_4_10 = 0.0

            for j in range(k+1):
                if MOS[id_out[j]] >= MOS[id_MOS[4]]:
                    temp_acc_4_5 += 1.0
                if MOS[id_out[j]] >= MOS[id_MOS[9]]:
                    temp_acc_4_10 += 1.0

            acc4_5[k] += temp_acc_4_5 / (k+1.0)
            acc4_10[k] += temp_acc_4_10 / (k+1.0)

        rank_of_returned_crop = []
        for k in range(4):
            rank_of_returned_crop.append(id_MOS.index(id_out[k]))

        for k in range(4):
            temp_wacc_4_5 = 0.0
            temp_wacc_4_10 = 0.0
            temp_rank_of_returned_crop = rank_of_returned_crop[:(k+1)]
            temp_rank_of_returned_crop.sort()
            for j in range(k+1):
                if temp_rank_of_returned_crop[j] <= 4:
                    temp_wacc_4_5 += 1.0 * \
                        math.exp(-0.2*(temp_rank_of_returned_crop[j]-j))
                if temp_rank_of_returned_crop[j] <= 9:
                    temp_wacc_4_10 += 1.0 * \
                        math.exp(-0.1*(temp_rank_of_returned_crop[j]-j))
            wacc4_5[k] += temp_wacc_4_5 / (k+1.0)
            wacc4_10[k] += temp_wacc_4_10 / (k+1.0)

        MOS_arr = []
        out = torch.squeeze(out).cpu().detach().numpy()
        for k in range(len(MOS)):
            MOS_arr.append(MOS[k].cpu().numpy()[0])

        srcc.append(spearmanr(MOS_arr, out)[0])
        pcc.append(pearsonr(MOS_arr, out)[0])

    for k in range(4):
        acc4_5[k] = acc4_5[k] / img_num
        acc4_10[k] = acc4_10[k] / img_num
        wacc4_5[k] = wacc4_5[k] / img_num
        wacc4_10[k] = wacc4_10[k] / img_num

    avg_srcc = sum(srcc) / img_num
    avg_pcc = sum(pcc) / img_num

    return acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description='Grid anchor based image cropping')
    parser.add_argument('--dataset_root', default='/public/datasets/CPCDataset/images',
                        help='Dataset root directory path')
    parser.add_argument('--base_model', default='mobilenetv2',
                        help='Pretrained base model')
    parser.add_argument('--scale', default='multi', type=str,
                        help='choose single or multi scale')
    parser.add_argument('--downsample', default=4,
                        type=int, help='downsample time')
    parser.add_argument('--augmentation', default=1, type=int,
                        help='choose single or multi scale')
    parser.add_argument('--image_size', default=256, type=int,
                        help='Batch size for training')
    parser.add_argument('--align_size', default=9, type=int,
                        help='Spatial size of RoIAlign and RoDAlign')
    parser.add_argument('--reduced_dim', default=8, type=int,
                        help='Spatial size of RoIAlign and RoDAlign')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4,
                        type=float, help='initial learning rate')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    args = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = build_crop_model(
        scale=args.scale, alignsize=args.align_size, reddim=args.reduced_dim,
        loadweight=True, model=args.base_model, downsample=args.downsample
    )

    if cuda:
        net = torch.nn.DataParallel(net, device_ids=[0])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.benchmark = True
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # 加载一个新的模型，继续训练

    model_dir = '/home/zhangshuo/paper/Grid-Anchor-based-Image-Cropping-Pytorch/pretrained_model/mobilenet_0.682_0.643_0.613_0.585_0.844_0.827_0.807_0.787_0.849_0.874.pth'

    new_state_dict = OrderedDict()
    state_dict = torch.load(model_dir)

    for k, v in state_dict.items():
        # 手动添加“module.”
        if 'module' not in k:
            k = 'module.' + k
        else:
            # 调换 module 和 features 的位置
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict)

    # 初始化存储累加结果的变量
    total_acc4_5 = None
    total_acc4_10 = None
    total_avg_srcc = 0  # 假设这是单个数值
    total_avg_pcc = 0   # 同上
    total_wacc4_5 = None
    total_wacc4_10 = None

    count = 0  # 用于计算平均值的计数器

    # 获取文件列表
    file_list = os.listdir('/home/zhangshuo/pic/datasets/test/5/val')

    # 循环遍历文件夹中的文件
    for user_name in tqdm(file_list, desc="处理进度", ascii=True):
        # 数据加载部分保持不变
        data_loader_test = data.DataLoader(
            CPCDataset(
                csv_file='/home/zhangshuo/pic/datasets/test/5/val/' + user_name,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='test'
            ),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        # 调用 test() 函数并获取结果
        acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10 = test()

        # 如果是第一次迭代，则初始化列表或变量
        if total_acc4_5 is None:
            total_acc4_5 = acc4_5
            total_acc4_10 = acc4_10
            total_wacc4_5 = wacc4_5
            total_wacc4_10 = wacc4_10
        else:
            total_acc4_5 = [x + y for x, y in zip(total_acc4_5, acc4_5)]
            total_acc4_10 = [x + y for x, y in zip(total_acc4_10, acc4_10)]
            total_wacc4_5 = [x + y for x, y in zip(total_wacc4_5, wacc4_5)]
            total_wacc4_10 = [x + y for x, y in zip(total_wacc4_10, wacc4_10)]

        # 累加单个数值
        total_avg_srcc += avg_srcc
        total_avg_pcc += avg_pcc

        count += 1
    print(count)
    # 计算列表的平均值
    mean_acc4_5 = [x / count for x in total_acc4_5]
    mean_acc4_10 = [x / count for x in total_acc4_10]
    mean_wacc4_5 = [x / count for x in total_wacc4_5]
    mean_wacc4_10 = [x / count for x in total_wacc4_10]

    # 计算单个数值的平均值
    mean_avg_srcc = total_avg_srcc / count
    mean_avg_pcc = total_avg_pcc / count

    # 定义一个输出列表的函数
    def print_mean(label, data):
        data_str = '\t'.join('{:.4f}'.format(num) for num in data)
        print(f"平均 {label}:\t{data_str}")

    # 输出每个列表中的数字，并保证每个数字只有四位小数
    print_mean("acc4_5", mean_acc4_5)
    print_mean("acc4_10", mean_acc4_10)
    print_mean("wacc4_5", mean_wacc4_5)
    print_mean("wacc4_10", mean_wacc4_10)

    print("平均 avg_srcc: {:.4f}".format(mean_avg_srcc))

    print("平均 avg_pcc: {:.4f}".format(mean_avg_pcc))
