# Personalitized Image Cropping

## Total

整个代码分为三个部分：pre, prior and finetune。每个部分可以选择的参数为：attention, gen and per.

```
pic
├─ datasets                                  // prior 与 finetune 阶段的少量数据，
│  ├─ csv_total
│  ├─ prior                                  // prior 阶段的数据
│  ├─ test                                   // finetune 阶段的数据
│  └─ tools                                  // 工具类，主要进行数据的自动划分
|
├─ dataloader
│  ├─ augmentations.py
│  ├─ finetune_cpc.py
│  ├─ gaic_cpc.py
│  ├─ pre_cpc.py
│  ├─ pre_cpc_best.py
│  ├─ pre_gaic.py
│  └─ prior_cpc.py
|
├─ finetune_cuhk.py                         // pic 方法在 cuhk 数据集上的微调测试
|
├─ finetune_gaic.py                         // gaic 方法在 cpc 数据集上的微调测试
|
├─ finetune_pic.py                          // pic 方法在 cpc 数据集上的微调测试
|
├─ gaic_cpc.py                              // gaic 方法在 cpc 数据集上的直接测试
|
├─ models
│  ├─ __init__.py
│  ├─ finetune_model.py                     // pic 方法 finetune阶段的模型
│  ├─ gaic_model.py                         // gaic 方法的基础模型
│  ├─ pre_model.py                          // pic 方法 在 pre 阶段的模型
│  ├─ pretrained_model                      // 预训练模型及代码
│  │  ├─ ShuffleNetV2.py
│  │  └─ mobilenetv2.py
│  ├─ prior_model.py                        // pic 方法在 prior 阶段的模型
│  └─ readme.md
|
├─ pre_pic.py                               // pic 方法的 pic 阶段代码
|
├─ prior_pic.py                             // pic 方法的 prior 阶段代码
|
├─ README.md
|
└─ tools                                    // Tools, 包括损失函数，roi与rod方法
   ├─ loss.py
   ├─ m_Loss.py
   ├─ rod_align
   └─ roi_align

```

## Pre

Pre 为预训练的过程，其中包含两个分支，客观分支参考了 GAIC 的代码，读取图片的 24 个裁剪框，并经过 RoI 与 RoD Align 操作；主观分支读取每位用户的最好三个裁剪框，并执行一个分类任务。

## Prior

Prior 为先验过程，提取构建一些用户图片组，对每位用户进行微调与单独实验，最终得到一个包含大量用户先验知识的裁剪模型。

## Finetune

Finutune 阶段为微调过程，对某一为用户的数据进行调整，并计算其指标。finetune\_\*的文件包含了针对不同数据与方法的代码，详见上面的文件目录。
