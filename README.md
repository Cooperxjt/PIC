# Personalitized Image Cropping

A code of Personalitized Image Cropping.

## Total

The whole code is divided into three parts: pre, prior and finetune.

```
pic
├─ dataloader
│  ├─ augmentations.py
│  ├─ finetune_cpc.py
│  ├─ gaic_cpc.py
│  ├─ pre_cpc.py
│  ├─ pre_cpc_best.py
│  ├─ pre_gaic.py
│  └─ prior_cpc.py
|
├─ finetune_cuhk.py                         // Fine-tuning test of the pic method on the cuhk dataset
|
├─ finetune_gaic.py                         // Fine-tuning test of the GAIC method on the cpc dataset
|
├─ finetune_pic.py                          // Fine-tuning test of the pic method on the cpc dataset
|
├─ gaic_cpc.py                              // GAIC method on the cpc dataset
|
├─ models
│  ├─ __init__.py
│  ├─ finetune_model.py                     // model of pic finetuning
│  ├─ gaic_model.py                         // model of gaic
│  ├─ pre_model.py                          // model of pic pre-traning
│  ├─ pretrained_model                      // pre-trained model
│  │  ├─ ShuffleNetV2.py
│  │  └─ mobilenetv2.py
│  ├─ prior_model.py                        // model of prior pic model
│  └─ readme.md
|
├─ pre_pic.py                               // pre-traning pic
|
├─ prior_pic.py                             // prior pic
|
├─ README.md
|
└─ tools
   ├─ loss.py
   ├─ m_Loss.py
   ├─ rod_align
   └─ roi_align

```

## Pre

Pre is the pre-training process, which consists of two branches: the objective branch refers to GAIC's code, which reads 24 cropped frames of the image and undergoes the RoI and RoD Align operations; and the subjective branch reads the best three cropped frames of each user and performs a classification task.

## Prior

Prior for the a priori process, extracting and constructing some groups of user pictures, fine-tuning and experimenting with each user individually, and finally obtaining a cropping model that contains a large amount of users' prior knowledge.

## Finetune

The Finutune phase is the fine-tuning process, where the data for a particular for user is adjusted and its metrics are calculated.
