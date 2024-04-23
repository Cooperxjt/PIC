import torch


def ensure_batch_dimension(tensor):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # 将维度从 [N] 变为 [1, N]
    return tensor


def scores_to_ranks(scores):
    # 确保scores是一个PyTorch张量
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)

    # 对分数进行降序排序并获取索引
    sorted_scores, indices = scores.sort(descending=True)

    # 初始化一个与scores相同长度的排名张量
    ranks = torch.zeros_like(scores, dtype=torch.float)

    # 处理相同分数的情况
    unique_scores, counts = torch.unique(sorted_scores, return_counts=True)
    for score, count in zip(unique_scores, counts):
        # 获取当前分数的索引
        idxs = (sorted_scores == score).nonzero(as_tuple=True)[0]
        # 计算平均排名
        avg_rank = idxs.float().mean() + 1  # 加1因为排名是从1开始的
        ranks[idxs] = avg_rank

    # 将排名映射回原始的顺序
    return ranks[indices.argsort()]


def loss_rank(predicted_list, user_score_list, avg_score_list):
    loss = 0

    # 确保 predicted_score 总是有两个维度
    predicted_list = ensure_batch_dimension(predicted_list)

    for idx, predicted_score in enumerate(predicted_list):

        # 将分数转换为排名
        predicted_ranks = scores_to_ranks(predicted_score)
        user_score_ranks = scores_to_ranks(user_score_list[idx])
        avg_score_ranks = scores_to_ranks(avg_score_list[idx])

        # print(predicted_ranks)
        # print(user_score_ranks)
        # print(avg_score_ranks)

        # 计算排名差异
        predicted_diff = predicted_ranks - avg_score_ranks
        user_score_diff = user_score_ranks - avg_score_ranks

        # print(predicted_diff)
        # print(user_score_diff)

        # 计算损失
        loss += torch.mean((user_score_diff - predicted_diff) ** 2)

    # print(f"排名差异损失: {loss.item()/len(predicted_list)}")
    return loss / len(predicted_list)
