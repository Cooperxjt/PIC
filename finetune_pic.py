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

from dataloader.finetune_cpc import CPCDataset_pic
from models.finetune_model import build_crop_model
from tools.m_Loss import loss_m
import csv


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (torch.Tensor): Tensor of shape (4,), representing [x1, y1, x2, y2].
        box2 (torch.Tensor): Tensor of shape (4,), representing [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    # Compute the area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the union area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def calculate_averages(results):

    total_acc4_5 = [0] * 4
    total_acc4_10 = [0] * 4
    total_avg_srcc = 0
    total_avg_pcc = 0
    total_wacc4_5 = [0] * 4
    total_wacc4_10 = [0] * 4

    for user, user_results in results.items():
        for i in range(4):
            total_acc4_5[i] += user_results['acc4_5'][i]
            total_acc4_10[i] += user_results['acc4_10'][i]
            total_wacc4_5[i] += user_results['wacc4_5'][i]
            total_wacc4_10[i] += user_results['wacc4_10'][i]

        total_avg_srcc += user_results['avg_srcc']
        total_avg_pcc += user_results['avg_pcc']

    num_users = len(results)

    avg_acc4_5 = [x / num_users * 100 for x in total_acc4_5]
    avg_acc4_10 = [x / num_users * 100 for x in total_acc4_10]
    avg_wacc4_5 = [x / num_users * 100 for x in total_wacc4_5]
    avg_wacc4_10 = [x / num_users * 100 for x in total_wacc4_10]
    avg_srcc = total_avg_srcc / num_users
    avg_pcc = total_avg_pcc / num_users

    return {
        'avg_acc4_5': avg_acc4_5,
        'avg_acc4_10': avg_acc4_10,
        'avg_srcc': avg_srcc,
        'avg_pcc': avg_pcc,
        'avg_wacc4_5': avg_wacc4_5,
        'avg_wacc4_10': avg_wacc4_10
    }


def test(epoch):
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
        # Construct two different types of data, belonging to generic and personality networks respectively
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

        # Image input obtained with another data enhancement applicable to personality networks
        image_cpc = sample['cpc_image']

        input_cpc = image_cpc.cuda()

        out = net(
            input_gaic, input_cpc
        )

        id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)

        real_box = roi_gaic[id_MOS[0]][1:5]
        pred_box = roi_gaic[id_out[0]][1:5]

        if epoch == 9:
            iou = calculate_iou(real_box, pred_box)
            iou_list.append(iou.item())

        # 计算 Acc
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

    best_acc4_5 = []
    best_acc4_10 = []
    best_avg_srcc = 0
    best_avg_pcc = 0
    best_wacc4_5 = []
    best_wacc4_10 = []

    for epoch in range(0, epoch_total):

        net.train()

        total_loss = 0
        loss = 0

        for sample in dataloader_finetune:

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

            loss = loss_sort + 0.8*loss_l1

            total_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10 = test(epoch)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    if not os.path.exists('user_weights/' + user_name):
        os.makedirs('user_weights/' + user_name)

    torch.save(
        net.state_dict(), 'user_weights/' +
        user_name + '/' + timestamp + '.pth'
    )

    if best_acc4_5 == [] or acc4_5[0] > best_acc4_5[0]:
        best_acc4_5 = acc4_5
        best_acc4_10 = acc4_10
        best_avg_srcc = avg_srcc
        best_avg_pcc = avg_pcc
        best_wacc4_5 = wacc4_5
        best_wacc4_10 = wacc4_10

    return best_acc4_5, best_acc4_10, best_avg_srcc, best_avg_pcc, best_wacc4_5, best_wacc4_10


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description='Grid anchor based image cropping')
    parser.add_argument('--model_dir', default='',
                        help='prior-model dir')
    parser.add_argument('--user_path', default='',
                        help='Path to user information')
    parser.add_argument('--dataset_root', default='',
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
    parser.add_argument('--lr', '--learning-rate', default=1e-3,
                        type=float, help='initial learning rate')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--N', default='5',
                        help='Num of images from each user')
    parser.add_argument('--gen', default=True,
                        help='Whether to enable generic branching')
    parser.add_argument('--per', default=True,
                        help='Whether to enable personalised branching')
    parser.add_argument('--attention', default=True,
                        help='Whether to turn on the attention mechanism')
    args = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    results = {}
    epoch_total = 10

    # Number of images per user
    N = args.N

    model_dir = args.model_dir

    print('load model：' + model_dir)

    print('epoch:' + str(epoch_total))
    print('N=' + str(N))

    gen = args.gen
    per = args.per
    attention = args.attention

    iou_list = []

    user_path = args.user_path

    for user_name in tqdm(os.listdir(user_path)):

        # 对一个用户提取其图片
        user_name = user_name[:-4]
        csv_file_train = user_path + str(N) + '/train/' + \
            user_name + '_train.csv'
        csv_file_val = user_path + str(N) + '/val/' + \
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

        net = build_crop_model(
            alignsize=args.align_size,
            reddim=args.reduced_dim,
            loadweight=True,
            model=args.base_model,
            downsample=args.downsample,
            num_classes=945,
            attention=attention,
            gen=gen,
            per=per,
        )

        state_dict = torch.load(model_dir)

        net.load_state_dict(state_dict, strict=False)

        if cuda:
            # net = torch.nn.DataParallel(net, device_ids=[0, 1])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            net = net.cuda()

        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        acc4_5, acc4_10, avg_srcc, avg_pcc, wacc4_5, wacc4_10 = finetune(
            epoch_total
        )

        results[user_name] = {
            'acc4_5': acc4_5,
            'acc4_10': acc4_10,
            'avg_srcc': avg_srcc,
            'avg_pcc': avg_pcc,
            'wacc4_5': wacc4_5,
            'wacc4_10': wacc4_10
        }

    avg_results = calculate_averages(results)

    def print_mean(label, data):
        print(f"平均 {label}: {' '.join('{:.4f}'.format(num) for num in data)}")

    print_mean("acc4_5", avg_results['avg_acc4_5'])
    print_mean("acc4_10", avg_results['avg_acc4_10'])
    print_mean("wacc4_5", avg_results['avg_wacc4_5'])
    print_mean("wacc4_10", avg_results['avg_wacc4_10'])
    print("average avg_srcc: {:.4f}".format(avg_results['avg_srcc']))
    print("average avg_pcc: {:.4f}".format(avg_results['avg_pcc']))

    # output_file = 'iou_results.csv'

    # with open(output_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['IoU'])
    #     for iou in iou_list:
    #         writer.writerow([iou])
