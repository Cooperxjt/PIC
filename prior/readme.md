训练一个 prior 模型，其中每一个 task 就是每个人的图片预测，目的是最终让模型可以输出一个人的裁剪排名

pic_model_best_new.py 在 pic 模型的基础上，额外增加了一个卷积层

prior_finetune.py 基于数据微调，得到 prior 模型，相较于之前的模型只增加了一层模型
