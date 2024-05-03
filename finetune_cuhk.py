# 在cuhk数据集上，根据每位用户的数据进行微调，并得到实验结果。
import argparse
import pickle
import random
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.finetune_cuhk_datasets import CUHKDataset_pic
from models.finetune_cuhk_model import build_crop_model
from tools.m_Loss import loss_m

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0*")


def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    """'
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    """

    gt_crop = gt_crop[gt_crop[:, 0] >= 0]

    zero_t = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:, 0], pre_crop[:, 0])
    over_y1 = torch.maximum(gt_crop[:, 1], pre_crop[:, 1])
    over_x2 = torch.minimum(gt_crop[:, 2], pre_crop[:, 2])
    over_y2 = torch.minimum(gt_crop[:, 3], pre_crop[:, 3])
    over_w = torch.maximum(zero_t, over_x2 - over_x1)
    over_h = torch.maximum(zero_t, over_y2 - over_y1)

    inter = over_w * over_h

    area1 = (gt_crop[:, 2] - gt_crop[:, 0]) * (gt_crop[:, 3] - gt_crop[:, 1])
    area2 = (pre_crop[:, 2] - pre_crop[:, 0]) * \
        (pre_crop[:, 3] - pre_crop[:, 1])

    union = area1 + area2 - inter
    iou = inter / union

    disp = (
        (
            torch.abs(gt_crop[:, 0] - pre_crop[:, 0])
            + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])
        ) / im_w + (
            torch.abs(gt_crop[:, 1] - pre_crop[:, 1])
            + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])
        ) / im_h) / 4

    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx

    return iou[index].item(), disp[index].item()


def test():
    net.eval()

    total_iou = 0
    total_disp = 0

    for sample in tqdm(dataloader_val):
        # 首先获取最好的裁剪结果
        gt_crop = [
            sample['ori_bbox']['xmin'][0], sample['ori_bbox']['ymin'][0],
            sample['ori_bbox']['xmax'][0], sample['ori_bbox']['ymax'][0]
        ]

        best_score = float('-inf')
        best_pred_bbox = None

        for batch_start in range(0, len(sample['bbox']['xmin']), 24):
            # 限制每批处理的最大裁剪框数量为24个
            batch_end = min(batch_start + 24, len(sample['bbox']['xmin']))

            if batch_end - batch_start < 24:  # 如果裁剪框数量不足24个
                # 重复已有裁剪框填充到24个
                repeat_count = 24 - (batch_end - batch_start)
                indices = list(range(batch_start, batch_end)) + \
                    [batch_start] * repeat_count  # 假设重复第一个裁剪框填充
            else:
                indices = list(range(batch_start, batch_end))

            # 构建两种不同类型的数据，分别属于通用网络和个性网络
            image_gaic = sample['image']

            # 根据indices来选取裁剪框
            bboxs_gaic = {
                k: [v[idx] for idx in indices]
                for k, v in sample['bbox'].items()
            }
            ori_bboxes = {
                k: [v[idx] for idx in indices]
                for k, v in sample['ori_bbox'].items()
            }

            MOS = sample['MOS'][batch_start:batch_end]
            img_width = sample['img_width']
            img_height = sample['img_height']

            roi_gaic = []
            ori_bboxes_list = []

            for idx in range(len(bboxs_gaic['xmin'])):
                roi_gaic.append(
                    (
                        0,
                        bboxs_gaic['xmin'][idx], bboxs_gaic['ymin'][idx],
                        bboxs_gaic['xmax'][idx], bboxs_gaic['ymax'][idx]
                    )
                )

                ori_bboxes_list.append(
                    (
                        0,
                        ori_bboxes['xmin'][idx], ori_bboxes['ymin'][idx],
                        ori_bboxes['xmax'][idx], ori_bboxes['ymax'][idx]
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

            # 更新最高得分和对应的裁剪框
            max_score_idx = out.argmax()

            if out[max_score_idx] > best_score:
                best_score = out[max_score_idx]
                best_pred_bbox = ori_bboxes_list[max_score_idx]

        best_pred_bbox = torch.tensor(
            [[x.item() for x in best_pred_bbox[1:5]]]
        )
        gt_crop = torch.tensor([[x.item() for x in gt_crop]])

        iou, disp = compute_iou_and_disp(
            gt_crop, best_pred_bbox, img_width, img_height
        )

        total_iou += iou
        total_disp += disp

    avg_iou = total_iou / len(dataloader_val)
    avg_disp = total_disp / len(dataloader_val)

    return avg_iou, avg_disp


def finetune(epoch_total):

    net.train()

    best_iou = 0
    best_disp = 10

    iou, disp = test()

    print(iou)
    print(disp)

    for epoch in range(0, epoch_total):
        total_loss = 0
        loss = 0
        # 创建迭代器

        for sample in dataloader_finetune:
            # 构建两种不同类型的数据，分别属于通用网络和个性网络
            image_gaic = sample['image']
            bboxs_gaic = sample['bbox']

            for batch_start in range(0, len(sample['bbox']['xmin']), 24):
                roi_gaic = []
                MOS_gaic = []

                # 限制每批处理的最大裁剪框数量为24个
                batch_end = min(batch_start + 24, len(sample['bbox']['xmin']))

                if batch_end - batch_start < 24:  # 如果裁剪框数量不足24个
                    # 重复已有裁剪框填充到24个
                    repeat_count = 24 - (batch_end - batch_start)
                    indices = list(range(batch_start, batch_end)) + \
                        [batch_start] * repeat_count  # 假设重复第一个裁剪框填充
                else:
                    indices = list(range(batch_start, batch_end))
                    # 根据indices来选取裁剪框
                    bboxs_gaic = {
                        k: [v[idx] for idx in indices]
                        for k, v in sample['bbox'].items()
                    }

                random_ID = list(range(0, len(bboxs_gaic['xmin'])))
                random.shuffle(random_ID)

                for idx in random_ID:
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

                loss = loss_l1 + 0.8 * loss_sort

                total_loss += loss.item()

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        iou, disp = test()

        print(f'Epoch: {epoch}, Iou: {iou}, Disp: {disp}')

        if best_iou == 0 or iou > best_iou:
            best_iou = iou
        if best_disp == 0 or disp < best_disp:
            best_disp = disp

    return best_iou, best_disp


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description='Grid anchor based image cropping')
    parser.add_argument('--dataset_root', default='/public/datasets/cuhk_cropping/All_Images',
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
    epoch_total = 50
    n = 50

    # 每位用户提供的图片数量

    # 加载网络
    model_dir = 'weights/prior/2024-03-22_02_09_54.pth'

    print('模型加载：' + model_dir)
    print('epoch:' + str(epoch_total))
    print('n:' + str(n))

    gen = args.gen
    per = args.per
    attention = args.attention

    # anchors = get_pdefined_anchors('dataloader/pdefined_anchor.pkl')

    for expert_id in range(1, 4):

        # 对一个用户提取其图片
        val_file = f'/public/datasets/cuhk_cropping/expert_{expert_id}_csv/{n}/expert_{expert_id}_val.csv'
        finetune_file = f'/public/datasets/cuhk_cropping/expert_{expert_id}_csv/{n}/expert_{expert_id}_finetune.csv'

        # 用于返回图片
        dataloader_finetune = data.DataLoader(
            CUHKDataset_pic(
                csv_file=finetune_file,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='train'
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            generator=torch.Generator(device='cuda')
        )

        dataloader_val = data.DataLoader(
            CUHKDataset_pic(
                csv_file=val_file,
                image_size=args.image_size,
                dataset_dir=args.dataset_root,
                set='val'
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            generator=torch.Generator(device='cuda')
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

        iou, disp = finetune(
            epoch_total
        )

        print(f'Expert{expert_id} Iou: {iou}, Disp: {disp}')

    # 输出 IoU Disp
