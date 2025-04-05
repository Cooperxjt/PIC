import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader.pre_cpc import CPC
from dataloader.pre_gaic import GAICD
from models.pre_model import build_crop_model


def train():
    net.train()

    loss_cpc = torch.zeros(1)
    loss_gaic = torch.zeros(1)

    for epoch in range(0, epoch_num):

        total_loss = 0
        total_gaic_loss = 0
        total_cpc_loss = 0

        bs_id = 0

        iterator_cpc = iter(data_loader_train_cpc)

        for sample_gaic in data_loader_train_gaic:
            # ---------------------------------------------------------#
            # --------------------       GEN       --------------------#
            # ---------------------------------------------------------#
            if gen == True:

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

                loss_gaic = torch.nn.SmoothL1Loss(reduction='mean')(
                    out_gaic, MOS_gaic
                )

            # ---------------------------------------------------------#
            # ----------------------      PER     ---------------------#
            # ---------------------------------------------------------#
            if per == True:
                try:
                    sample_cpc = next(iterator_cpc)
                except StopIteration:
                    iterator_cpc = iter(data_loader_train_cpc)
                    sample_cpc = next(iterator_cpc)

                input_cpc = sample_cpc[0].cuda()
                labels_cpc = sample_cpc[1].cuda()

                out_cpc = net('cpc', input_cpc)

                loss_cpc = torch.nn.CrossEntropyLoss()(out_cpc, labels_cpc)

                total_cpc_loss += loss_gaic.item()
                total_gaic_loss += loss_cpc.item()

            loss = loss_gaic + loss_cpc

            total_loss += loss.item()

            avg_loss = total_loss / (bs_id+1)
            avg_cpc_loss = total_cpc_loss / (bs_id+1)
            avg_gaic_loss = total_gaic_loss / (bs_id+1)

            bs_id += 1

            writer.add_scalar('Loss/GAIC', loss_gaic.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/CPC', loss_cpc.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/Total', loss.item(),
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/avg', avg_loss,
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/avg_cpc', avg_cpc_loss,
                              epoch * len(data_loader_train_gaic) + bs_id)
            writer.add_scalar('Loss/avg_gaic', avg_gaic_loss,
                              epoch * len(data_loader_train_gaic) + bs_id)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [Train Loss: %.4f]' %
                (epoch, epoch_num, bs_id, len(data_loader_train_gaic), avg_loss)
            )

        torch.save(
            net.state_dict(), args.save_folder +
            '/' + repr(epoch) + '.pth'
        )


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description='Grid anchor based image cropping')
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
    parser.add_argument('--lr', '--learning-rate', default=1e-4,
                        type=float, help='initial learning rate')
    parser.add_argument('--save_folder', default='weights/pre/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--gen', default=True,
                        help='Whether to enable generic branching')
    parser.add_argument('--per', default=True,
                        help='Whether to enable personalised branching')
    parser.add_argument('--attention', default=True,
                        help='Whether or not to switch on the attention mechanism')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    args.save_folder = args.save_folder + '/' + timestamp

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    print(args.save_folder)

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    print("load net:" + args.base_model)
    gaic_dataset = GAICD(
        set='train',
        image_size=args.image_size,
        dataset_dir=args.dataset_root,
        augmentation=args.augmentation
    )

    data_loader_train_gaic = data.DataLoader(
        gaic_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=random.seed(SEED),
        generator=torch.Generator(device='cuda')
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.RandomCrop(256, padding=8, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]
        )
    ])

    cpc_dataset = CPC(
        main_dir='',
        transform=transform
    )

    data_loader_train_cpc = data.DataLoader(
        cpc_dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )

    num_classes = len(cpc_dataset.classes)

    gen = args.gen
    per = args.per
    attention = args.attention

    net = build_crop_model(
        alignsize=args.align_size,
        reddim=args.reduced_dim,
        loadweight=True,
        model=args.base_model,
        downsample=args.downsample,
        num_classes=num_classes,
        attention=attention,
        gen=gen,
        per=per,
    )

    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint)
        print(f"Model loaded from checkpoint: {args.resume}")

    if cuda:
        # net = torch.nn.DataParallel(net, device_ids=[0, 1])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('runs/cpc_best', timestamp)
    writer = SummaryWriter(log_dir)

    epoch_num = 200

    train()
