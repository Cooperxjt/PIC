import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from fightingcv_attention.attention.DANet import DAModule

from models.mobilenetv2 import MobileNetV2
from models.rod_align.modules.rod_align import RoDAlign, RoDAlignAvg
from models.roi_align.modules.roi_align import RoIAlign, RoIAlignAvg
from models.ShuffleNetV2 import shufflenetv2


class vgg_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(vgg_base, self).__init__()

        vgg = models.vgg16(pretrained=True)

        if downsample == 4:
            self.feature = nn.Sequential(vgg.features[:-1])
        elif downsample == 5:
            self.feature = nn.Sequential(vgg.features)

        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])

        # flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        # return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class resnet50_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(resnet50_base, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.feature3 = nn.Sequential(
            resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2)
        self.feature4 = nn.Sequential(resnet50.layer3)
        self.feature5 = nn.Sequential(resnet50.layer4)

        # flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        # return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class mobilenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='/home/zhangshuo/pic/models/pretrained_model/mobilenetv2_1.0-0c6065bc.pth'):
        super(mobilenetv2_base, self).__init__()

        model = MobileNetV2(width_mult=1.0)

        if loadweights:
            model.load_state_dict(torch.load(model_path))

        # if downsample == 4:
        #    self.feature = nn.Sequential(model.features[:14])
        # elif downsample == 5:
        #    self.feature = nn.Sequential(model.features)

        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:])

        # flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        # return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class shufflenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='/home/zhangshuo/pic/models/pretrained_model/shufflenetv2_x1_69.402_88.374.pth.tar'):
        super(shufflenetv2_base, self).__init__()

        model = shufflenetv2(width_mult=1.0)

        if loadweights:
            model.load_state_dict(torch.load(model_path))

        self.feature3 = nn.Sequential(
            model.conv1, model.maxpool, model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:12])
        self.feature5 = nn.Sequential(model.features[12:])

        # if downsample == 4:
        #    self.feature = nn.Sequential(model.conv1, model.maxpool, model.features[:12])
        # elif downsample == 5:
        #    self.feature = nn.Sequential(model.conv1, model.maxpool, model.features)

        # flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        # return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


def fc_layers_1(reddim=32, alignsize=8):
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize,
                          padding=0), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1),
                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))
    dropout = nn.Dropout(p=0.5)
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, dropout, conv3)

    return layers


def fc_layers_2(reddim=32, num_classes=1345, alignsize=8):
    conv1 = nn.Sequential(
        nn.Conv2d(
            reddim, 768, kernel_size=alignsize,
            padding=0
        ),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True)
    )
    conv2 = nn.Sequential(
        nn.Conv2d(768, 128, kernel_size=1),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True)
    )
    dropout = nn.Dropout(p=0.5)
    conv3 = nn.Conv2d(128, num_classes, kernel_size=1)
    global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化层
    flatten = nn.Flatten()  # 将池化后的 1x1xnum_classes 张量展平

    layers = nn.Sequential(
        conv1, conv2, dropout, conv3,
        global_avg_pool, flatten
    )

    return layers


class persionality_crop_model_multi_scale_shared(nn.Module):

    def __init__(self, alignsize=8, reddim=32, loadweight=True, model=None, downsample=4, num_classes=1345):
        super(persionality_crop_model_multi_scale_shared, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(3584, reddim, kernel_size=1, padding=0)

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)

        self.FC_layers_1 = fc_layers_1(reddim*2, alignsize)
        self.FC_layers_2 = fc_layers_2(8, num_classes, alignsize)

    def forward(self, label, data):
        if label == 'gaic':
            # ---------------------------------------------------------#
            # ----------------------处理 gaic 的数据--------------------#
            # ---------------------------------------------------------#
            im_data_gaic = data['image_gaic']
            boxes_gaic = data['roi_gaic']

            f3, f4, f5 = self.Feat_ext(im_data_gaic)

            cat_feat = torch.cat(
                (self.downsample2(f3), f4, 0.5*self.upsample2(f5)), 1)

            red_feat = self.DimRed(cat_feat)

            RoI_feat = self.RoIAlign(red_feat, boxes_gaic)
            RoD_feat = self.RoDAlign(red_feat, boxes_gaic)

            final_feat = torch.cat((RoI_feat, RoD_feat), 1)

            print(final_feat.shape)

            prediction_1 = self.FC_layers_1(final_feat)

            return prediction_1.squeeze()

        # ---------------------------------------------------------#
        # ----------------------处理 cpc 的数据---------------------#
        # ---------------------------------------------------------#
        if label == 'cpc':
            im_data_cpc = data

            f3, f4, f5 = self.Feat_ext(im_data_cpc)

            cat_feat = torch.cat(
                (self.downsample2(f3), f4, 0.5*self.upsample2(f5)), 1)

            red_feat = self.DimRed(cat_feat)

            print(red_feat.shape)

            prediction_2 = self.FC_layers_2(red_feat)

            return prediction_2.squeeze()

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


class persionality_crop_model_multi_scale_shared_attention(nn.Module):

    def __init__(self, alignsize=8, reddim=32, loadweight=True, model=None, downsample=4, num_classes=1345, gen=True, per=True):
        super(persionality_crop_model_multi_scale_shared_attention, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
            self.danet = DAModule(d_model=812, kernel_size=3, H=16, W=16)

        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
            self.danet = DAModule(d_model=448, kernel_size=3, H=16, W=16)

        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)

        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(3584, reddim, kernel_size=1, padding=0)

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)

        self.FC_layers_1 = fc_layers_1(reddim*2, alignsize)
        self.FC_layers_2 = fc_layers_2(8, num_classes, alignsize)

        self.gen = gen
        self.per = per

    def forward(self, label, data):
        if label == 'gaic' and self.gen == True:
            # ---------------------------------------------------------#
            # ----------------------处理 gaic 的数据--------------------#
            # ---------------------------------------------------------#
            im_data_gaic = data['image_gaic']
            boxes_gaic = data['roi_gaic']

            f3, f4, f5 = self.Feat_ext(im_data_gaic)

            cat_feat = torch.cat(
                (self.downsample2(f3), f4, 0.5*self.upsample2(f5)), 1
            )

            red_feat = self.DimRed(cat_feat)

            RoI_feat = self.RoIAlign(red_feat, boxes_gaic)
            RoD_feat = self.RoDAlign(red_feat, boxes_gaic)

            final_feat = torch.cat((RoI_feat, RoD_feat), 1)

            prediction_1 = self.FC_layers_1(final_feat)

            return prediction_1.squeeze()

        if label == 'cpc' and self.per == True:
            # ---------------------------------------------------------#
            # ----------------------处理 cpc 的数据---------------------#
            # ---------------------------------------------------------#
            im_data_cpc = data

            f3, f4, f5 = self.Feat_ext(im_data_cpc)

            cat_feat = torch.cat(
                (self.downsample2(f3), f4, 0.5*self.upsample2(f5)), 1
            )

            da_feat = self.danet(cat_feat)

            red_feat = self.DimRed(da_feat)

            prediction_2 = self.FC_layers_2(red_feat)

            return prediction_2.squeeze()

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def build_crop_model(alignsize=8, reddim=32, loadweight=True, model=None, downsample=4, num_classes=1345, attention=False, gen=True, per=True):

    if attention:
        return persionality_crop_model_multi_scale_shared_attention(alignsize, reddim, loadweight, model, downsample, num_classes, gen, per)
    else:
        return persionality_crop_model_multi_scale_shared(alignsize, reddim, loadweight, model, downsample, num_classes)