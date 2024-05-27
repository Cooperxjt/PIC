import argparse
import math
import os
import random
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.gaic_cpc import CPCDataset
from models.gaic_model import build_crop_model

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0*")


def calculate_averages(results):
    # 初始化累加器
    total_acc4_5 = [0] * 4
    total_acc4_10 = [0] * 4
    total_avg_srcc = 0
    total_avg_pcc = 0
    total_wacc4_5 = [0] * 4
    total_wacc4_10 = [0] * 4

    # 累加每个用户的结果
    for user, user_results in results.items():
        for i in range(4):
            total_acc4_5[i] += user_results['acc4_5'][i]
            total_acc4_10[i] += user_results['acc4_10'][i]
            total_wacc4_5[i] += user_results['wacc4_5'][i]
            total_wacc4_10[i] += user_results['wacc4_10'][i]

        total_avg_srcc += user_results['avg_srcc']
        total_avg_pcc += user_results['avg_pcc']

    num_users = len(results)

    # 计算平均值
    avg_acc4_5 = [x / num_users * 100 for x in total_acc4_5]
    avg_acc4_10 = [x / num_users * 100 for x in total_acc4_10]
    avg_wacc4_5 = [x / num_users * 100 for x in total_wacc4_5]
    avg_wacc4_10 = [x / num_users * 100 for x in total_wacc4_10]
    avg_srcc = total_avg_srcc / num_users
    avg_pcc = total_avg_pcc / num_users

    # 返回平均结果
    return {
        'avg_acc4_5': avg_acc4_5,
        'avg_acc4_10': avg_acc4_10,
        'avg_srcc': avg_srcc,
        'avg_pcc': avg_pcc,
        'avg_wacc4_5': avg_wacc4_5,
        'avg_wacc4_10': avg_wacc4_10
    }


def test():
    net.eval()

    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    total_loss = 0
    avg_loss = 0

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
        loss = torch.nn.SmoothL1Loss(reduction='mean')(
            out.squeeze(), torch.Tensor(MOS))

        total_loss += loss.item()
        avg_loss = total_loss / (id+1)

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

    return acc4_5, acc4_10, avg_srcc, avg_pcc, avg_loss, wacc4_5, wacc4_10


def train():
    net.train()

    for epoch in range(0, 10):

        for id, sample in enumerate(data_loader_train):

            image = sample['image']
            bboxs = sample['bbox']

            roi = []
            MOS = []

            random_ID = list(range(0, len(bboxs['xmin'])))
            random.shuffle(random_ID)

            for idx in random_ID[:64]:
                roi.append(
                    (
                        0, bboxs['xmin'][idx], bboxs['ymin']
                        [idx], bboxs['xmax'][idx], bboxs['ymax'][idx]
                    )
                )
                MOS.append(sample['MOS'][idx])

            if cuda:
                image = Variable(image.cuda())
                roi = Variable(torch.Tensor(roi))
                MOS = torch.Tensor(MOS)
            else:
                image = Variable(image)
                roi = Variable(roi)

            # forward
            out = net(image, roi)

            loss = torch.nn.SmoothL1Loss(
                reduction='mean')(out.squeeze(), MOS)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10 = test()

    return acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10


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
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-4,
                        type=float, help='initial learning rate')
    args = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    results = {}

    # 初始化存储累加结果的变量
    total_acc4_5 = None
    total_acc4_10 = None
    total_avg_srcc = 0  # 假设这是单个数值
    total_avg_pcc = 0   # 同上
    total_wacc4_5 = None
    total_wacc4_10 = None

    for user_name in tqdm(os.listdir('datasets/finetune/total')):

        # 对一个用户提取其图片
        user_name = user_name[:-4]
        csv_file_train = 'datasets/finetune/5/train/' + \
            user_name + '_train.csv'
        csv_file_val = 'datasets/finetune/5/val/' + \
            user_name + '_val.csv'

        data_loader_train = data.DataLoader(
            CPCDataset(
                csv_file=csv_file_train,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='train',
                augmentation=args.augmentation),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=random.seed(SEED),
            generator=torch.Generator(device='cuda')
        )

        data_loader_test = data.DataLoader(
            CPCDataset(
                csv_file=csv_file_val,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='test'
            ),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )

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

        # 加载一个新的模型，继续训练
        resume_dir = 'models/pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth'

        new_state_dict = OrderedDict()
        state_dict = torch.load(resume_dir)

        for k, v in state_dict.items():
            # 手动添加“module.”
            if 'module' not in k:
                k = 'module.'+k
            else:
                # 调换module和features的位置
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        net.load_state_dict(new_state_dict)

        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10 = train()

        # 将结果存储到字典中
        results[user_name] = {
            'acc4_5': acc4_5,
            'acc4_10': acc4_10,
            'avg_srcc': avg_srcc,
            'avg_pcc': avg_pcc,
            'wacc4_5': wacc4_5,
            'wacc4_10': wacc4_10
        }

    # 计算平均值
    avg_results = calculate_averages(results)

    # 定义一个输出列表的函数
    def print_mean(label, data):
        print(f"平均 {label}: {' '.join('{:.4f}'.format(num) for num in data)}")

    # 输出每个列表中的数字，并保证每个数字只有四位小数
    print_mean("acc4_5", avg_results['avg_acc4_5'])
    print_mean("acc4_10", avg_results['avg_acc4_10'])
    print_mean("wacc4_5", avg_results['avg_wacc4_5'])
    print_mean("wacc4_10", avg_results['avg_wacc4_10'])
    print("平均 avg_srcc: {:.4f}".format(avg_results['avg_srcc']))
    print("平均 avg_pcc: {:.4f}".format(avg_results['avg_pcc']))
