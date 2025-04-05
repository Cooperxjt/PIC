import torch


def ensure_batch_dimension(tensor):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def scores_to_ranks(scores):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)

    sorted_scores, indices = scores.sort(descending=True)

    ranks = torch.zeros_like(scores, dtype=torch.float)

    # 处理相同分数的情况
    unique_scores, counts = torch.unique(sorted_scores, return_counts=True)
    for score, count in zip(unique_scores, counts):
        idxs = (sorted_scores == score).nonzero(as_tuple=True)[0]
        avg_rank = idxs.float().mean() + 1
        ranks[idxs] = avg_rank

    return ranks[indices.argsort()]


def loss_rank(predicted_list, user_score_list, avg_score_list):
    loss = 0

    predicted_list = ensure_batch_dimension(predicted_list)

    for idx, predicted_score in enumerate(predicted_list):

        predicted_ranks = scores_to_ranks(predicted_score)
        user_score_ranks = scores_to_ranks(user_score_list[idx])
        avg_score_ranks = scores_to_ranks(avg_score_list[idx])

        predicted_diff = predicted_ranks - avg_score_ranks
        user_score_diff = user_score_ranks - avg_score_ranks

        # 计算损失
        loss += torch.mean((user_score_diff - predicted_diff) ** 2)

    return loss / len(predicted_list)
