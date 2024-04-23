# 修改了迭代的方式，cpc用迭代器进行

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from dataloader.picDataset_cpc import CPC
from dataloader.picDataset_gaic import GAICD
from models.picModel_2 import build_crop_model
from tools.loss import loss_rank


def custom_collate_fn(batch):
    # 这里的实现需要根据您的具体数据结构进行调整
    collated_batch = {'user': [], 'images': []}
    for item in batch:
        collated_batch['user'].append(item['user'])
        collated_batch['images'].append(item['images'])
    return collated_batch


def test():
    data_loader_test_gaic = {}

    net.eval()
    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    total_loss = 0
    avg_loss = 0
    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for id, sample in enumerate(data_loader_test_gaic):
        image = sample['image']
        bboxs = sample['bbox']
        MOS = sample['MOS']

        roi = []

        for idx in range(0, len(bboxs['xmin'])):
            roi.append((0, bboxs['xmin'][idx], bboxs['ymin']
                       [idx], bboxs['xmax'][idx], bboxs['ymax'][idx]))

        if cuda:
            image = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            image = Variable(image)
            roi = Variable(roi)

        # t0 = time.time()
        out = net(image, roi)
        loss = torch.nn.SmoothL1Loss(reduction='elementwise_mean')(
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

        # t1 = time.time()

        # print('timer: %.4f sec.' % (t1 - t0))
    for k in range(4):
        acc4_5[k] = acc4_5[k] / 200.0
        acc4_10[k] = acc4_10[k] / 200.0
        wacc4_5[k] = wacc4_5[k] / 200.0
        wacc4_10[k] = wacc4_10[k] / 200.0

    avg_srcc = sum(srcc) / 200.0
    avg_pcc = sum(pcc) / 200.0

    return acc4_5, acc4_10, avg_srcc, avg_pcc, avg_loss, wacc4_5, wacc4_10


def train():

    net.train()

    for epoch in range(0, 80):
        total_loss = 0

        bs_id = 0

        # 创建迭代器
        iterator_cpc = iter(data_loader_train_cpc)

        for sample_gaic in data_loader_train_gaic:
            # ---------------------------------------------------------#
            # ----------------------处理 gaic 的数据--------------------#
            # ---------------------------------------------------------#
            image_gaic = sample_gaic['image']
            bboxs_gaic = sample_gaic['bbox']

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

                MOS_gaic.append(sample_gaic['MOS'][idx])

            if cuda:
                image_gaic = Variable(image_gaic.cuda())
                roi_gaic = Variable(torch.Tensor(roi_gaic))
                MOS_gaic = torch.Tensor(MOS_gaic)
            else:
                image_gaic = Variable(image_gaic)
                roi_gaic = Variable(roi_gaic)

            input_data = {
                'image_gaic': image_gaic,
                'roi_gaic': roi_gaic
            }

            out_gaic = net(
                'gaic', input_data
            )

            # loss_gaic 的通用裁剪 loss这里采用 SL1 loss。
            loss_gaic = torch.nn.SmoothL1Loss(reduction='mean')(
                out_gaic, MOS_gaic
            )

            # ---------------------------------------------------------#
            # ----------------------处理 cpc 的数据---------------------#
            # ---------------------------------------------------------#
            try:
                sample_cpc = next(iterator_cpc)
            except StopIteration:
                # 重启CPC迭代器
                iterator_cpc = iter(data_loader_train_cpc)
                sample_cpc = next(iterator_cpc)

            # 最终要输入该用户的所有图片的平均分数、打的分数以及偏差，我的目的是网络也得到打的分数，使得偏差与偏差之间尽可能地小。
            user = sample_cpc['user']

            # tensor_list 存储多张图片的 tensor, roi_cpc存储一个 roi 的列表, avg_score为图片的平均分数，user_score为图片的用户打分
            cpc_tensor_list = []
            roi_cpc = []
            avg_score_list = []
            user_score_list = []

            for image_data in sample_cpc['images']:
                for image_name, image_info in image_data.items():
                    image_cpc = image_info['image']
                    tensor_temp = torch.from_numpy(image_cpc)
                    cpc_tensor_list.append(tensor_temp)
                    ori_bboxes = image_info['ori_bboxes']
                    re_bboxes = image_info['transformed_bboxes']
                    user_score = image_info['user_scores']
                    avg_score = image_info['avg_scores']
                    score_diff = image_info['score_diffs']

                    roi_cpc_temp = []

                    for idx in range(len(re_bboxes['xmin'])):
                        roi_cpc_temp.append(
                            (
                                0,
                                re_bboxes['xmin'][idx], re_bboxes['ymin'][idx],
                                re_bboxes['xmax'][idx], re_bboxes['ymax'][idx]
                            )
                        )

                    roi_cpc.append(roi_cpc_temp)

                    avg_score_list.append(avg_score)
                    user_score_list.append(user_score)

            # 使用 torch.stack 沿着第0维度（新的维度）堆叠这些张量
            cpc_stacked_tensors = torch.stack(cpc_tensor_list).cuda()

            if cuda:
                cpc_stacked_tensors = Variable(cpc_stacked_tensors.cuda())
                roi_cpc = Variable(torch.Tensor(roi_cpc))
                user_score_list = torch.Tensor(user_score_list)
                avg_score_list = torch.Tensor(avg_score_list)
            else:
                cpc_stacked_tensors = Variable(cpc_stacked_tensors)
                roi_cpc = Variable(roi_cpc)

            input_data = {
                'image_cpc': cpc_stacked_tensors,
                'roi_cpc': roi_cpc
            }

            out_cpc = net(
                'cpc', input_data
            )

            # loss_cpc 计算序列之间的损失
            loss_cpc = loss_rank(out_cpc, user_score_list, avg_score_list)

            # 计算全部的 loss
            loss = loss_gaic + 0.01 * loss_cpc

            total_loss += loss.item()
            avg_loss = total_loss / (bs_id+1)
            bs_id += 1

            # 记录loss的变化
            writer.add_scalar('Loss/GAIC', loss_gaic.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/CPC', loss_cpc.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/Total', loss.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r[Epoch %d/%d] [Batch %d/%d] [Train Loss: %.4f]' %
                             (epoch, 79, bs_id, len(data_loader_train_gaic), avg_loss))


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description='Grid anchor based image cropping')
    parser.add_argument('--dataset_root', default='/public/datasets/GAIC_2/',
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
    parser.add_argument('--save_folder', default='weights/ablation/cropping/',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()

    args.save_folder = args.save_folder + args.base_model + '/' + 'downsample' + \
        str(args.downsample) + '_' + args.scale + '_Aug' + str(args.augmentation) + \
        '_Align' + str(args.align_size) + '_Cdim'+str(args.reduced_dim)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    data_loader_train_gaic = data.DataLoader(
        GAICD(
            set='train',
            image_size=args.image_size,
            dataset_dir=args.dataset_root,
            augmentation=args.augmentation
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=random.seed(SEED),
        generator=torch.Generator(device='cuda')
    )

    csv_folder = '/public/datasets/CPCDataset/process/user_score_process'
    txt_folder = '/public/datasets/CPCDataset/process/avg_results'
    img_folder = '/public/datasets/CPCDataset/images'

    data_loader_train_cpc = data.DataLoader(
        CPC(
            image_size=args.image_size,
            csv_folder=csv_folder,
            txt_folder=txt_folder,
            img_folder=img_folder,
            set='train'
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        generator=torch.Generator(device='cuda')
    )

    net = build_crop_model(
        alignsize=args.align_size,
        reddim=args.reduced_dim,
        loadweight=True,
        model=args.base_model,
        downsample=args.downsample
    )

    if cuda:
        # net = torch.nn.DataParallel(net, device_ids=[0, 1])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 创建TensorBoard写入器
    writer = SummaryWriter('runs/12_27')

    train()
