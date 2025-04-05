import argparse
import math
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.prior_cpc import CPCDataset_pic
from models.prior_model import build_crop_model
from tools.m_Loss import loss_m

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
    avg_acc4_10 = [x / num_users for x in total_acc4_10]
    avg_wacc4_5 = [x / num_users for x in total_wacc4_5]
    avg_wacc4_10 = [x / num_users for x in total_wacc4_10]
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

    img_num = len(dataloader_val)

    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for sample in dataloader_val:
        # 构建两种不同类型的数据，分别属于通用网络和个性网络
        image_gaic = sample['image']
        bboxs_gaic = sample['bbox']
        MOS = sample['MOS']

        roi_gaic = []

        for idx in range(len(bboxs_gaic['xmin'])):
            roi_gaic.append(
                (
                    0,
                    bboxs_gaic['xmin'][idx], bboxs_gaic['ymin'][idx],
                    bboxs_gaic['xmax'][idx], bboxs_gaic['ymax'][idx]
                )
            )

        if cuda:
            image_gaic = Variable(image_gaic.cuda())
            roi_gaic = Variable(torch.Tensor(roi_gaic))
        else:
            image_gaic = Variable(image_gaic)
            roi_gaic = Variable(roi_gaic)

        input_gaic = {
            'image_gaic': image_gaic,
            'roi_gaic': roi_gaic
        }

        # 用另一种数据增强得到的适用于个性网络的图片输入
        image_cpc = sample['cpc_image']

        input_cpc = image_cpc.cuda()

        out = net(
            input_gaic, input_cpc
        )

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


def finetune(epoch_total):

    net.train()

    best_acc4_5 = []
    best_acc4_10 = []
    best_avg_srcc = 0
    best_avg_pcc = 0
    best_wacc4_5 = []
    best_wacc4_10 = []

    for epoch in range(0, epoch_total):
        total_loss = 0
        loss = 0
        # 创建迭代器

        for sample in dataloader_finetune:
            # 构建两种不同类型的数据，分别属于通用网络和个性网络
            image_gaic = sample['image']
            bboxs_gaic = sample['bbox']

            roi_gaic = []
            MOS_gaic = []

            random_ID = list(range(0, len(bboxs_gaic['xmin'])))
            random.shuffle(random_ID)

            for idx in random_ID[:64]:
                roi_gaic.append(
                    (
                        0,
                        bboxs_gaic['xmin'][idx], bboxs_gaic['ymin'][idx],
                        bboxs_gaic['xmax'][idx], bboxs_gaic['ymax'][idx]
                    )
                )

                MOS_gaic.append(sample['MOS'][idx])

            if cuda:
                image_gaic = Variable(image_gaic.cuda())
                roi_gaic = Variable(torch.Tensor(roi_gaic))
                MOS_gaic = torch.Tensor(MOS_gaic)
            else:
                image_gaic = Variable(image_gaic)
                roi_gaic = Variable(roi_gaic)

            input_gaic = {
                'image_gaic': image_gaic,
                'roi_gaic': roi_gaic
            }

            # 用另一种数据增强得到的适用于个性网络的图片输入
            image_cpc = sample['cpc_image']

            input_cpc = image_cpc.cuda()

            out = net(
                input_gaic, input_cpc
            )

            loss_sort = loss_m(out, MOS_gaic).mean()

            # 计算全部的 loss
            loss_l1 = torch.nn.SmoothL1Loss(reduction='mean')(
                out, MOS_gaic
            )

            loss = 0.8 * loss_sort + loss_l1

            total_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10 = test()

        # if best_acc4_5 == [] or acc4_5[0] > best_acc4_5[0]:
        #     best_acc4_5 = acc4_5
        #     best_acc4_10 = acc4_10
        #     best_avg_srcc = avg_srcc
        #     best_avg_pcc = avg_pcc
        #     best_wacc4_5 = wacc4_5
        #     best_wacc4_10 = wacc4_10

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
    parser.add_argument('--base_model', default='resnet50',
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
    parser.add_argument('--lr', '--learning-rate', default=1e-3,
                        type=float, help='initial learning rate')
    parser.add_argument('--save_folder', default='weights/prior/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--gen', default=True,
                        help='是否开启通用分支')
    parser.add_argument('--per', default=True,
                        help='是否开启个性化分支')
    parser.add_argument('--attention', default=True,
                        help='是否开启注意力机制')
    args = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    results = {}

    # 加载网络
    model_dir = 'weights/pre/2024-03-18_10_07_19/130.pth'

    state_dict = torch.load(model_dir)

    gen = args.gen
    per = args.per
    attention = args.attention

    net = build_crop_model(
        alignsize=args.align_size,
        reddim=args.reduced_dim,
        loadweight=True,
        model=args.base_model,
        downsample=args.downsample,
        attention=attention,
        gen=gen,
        per=per,
    )

    net.load_state_dict(state_dict, strict=False)

    print('模型加载：' + model_dir)

    if cuda:
        # net = torch.nn.DataParallel(net, device_ids=[0, 1])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        net = net.cuda()

    epoch_total = 20

    print('epoch:' + str(epoch_total))

    for user_name in tqdm(os.listdir('datasets/prior/2/total')):

        # 对一个用户提取其图片
        user_name = user_name[:-4]
        csv_file_train = 'datasets/prior/2/5/train/' + \
            user_name + '_train.csv'
        csv_file_val = 'datasets/prior/2/5/val/' + \
            user_name + '_val.csv'

        # 用于返回图片
        dataloader_finetune = data.DataLoader(
            CPCDataset_pic(
                csv_file=csv_file_train,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='test'
            ),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )

        dataloader_val = data.DataLoader(
            CPCDataset_pic(
                csv_file=csv_file_val,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='test'
            ),
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )

        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        # 对于每一个人 finetune
        acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10 = finetune(
            epoch_total)

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
    print("平均 avg_srcc:{:.4f}".format(avg_results['avg_srcc']))
    print("平均 avg_pcc:{:.4f}".format(avg_results['avg_pcc']))

    timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    torch.save(net.state_dict(), args.save_folder + timestamp + '.pth')

    print(args.save_folder + timestamp + '.pth')
