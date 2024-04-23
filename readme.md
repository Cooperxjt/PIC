main_1.py: 一个 batch 训练两个数据集，小的数据集结束就结束

mian_2.py: 两个数据集同步训练，大的结束才结束，小的会多次迭代

main_best.py: 不再训练用户的全部图片，而是只训练用户的最满意构图，对应 PICDataset_cpc_best

模型分为三个部分，
第一阶段 pretrain，pre 阶段，gaic 正常训练，使用原始数据，cpc 则作为一个分类任务，每张图片只训练其 234 的图片，训练这些图片属于这个用于的分类任务
第二阶段 prior，用 cpc 的数据，每个人当做一个 task 进行训练，有两个目的，一个是让最终模型可以输出一个完整的预测，另一个是增加个性化的训练数据
第三阶段 test，用剩下的 cpc 数据进行测试

数据划分：
/public/datasets/CPCDataset/process/user_best_images_split 下面包含 train 和 test，test 为 400，train 为 395，每个里面包含每个人的 234 分图片
