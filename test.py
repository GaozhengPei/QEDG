import torch


def cosine_similarity(matrix):
    # 计算向量的内积
    dot_product = torch.matmul(matrix, matrix.t())

    # 计算向量的范数
    norm = torch.norm(matrix, dim=1, keepdim=True)

    # 计算余弦相似度
    similarity = dot_product / torch.matmul(norm, norm.t())

    return similarity


# 计算类内相似度的和
def sum_intra_similarity(matrix):
    similarity_matrix = cosine_similarity(matrix)
    # 选择除了对角线以上的元素
    mask = torch.triu(torch.ones(similarity_matrix.shape), diagonal=1)

    # 将对角线以上的元素置为0，然后计算总和
    intra_similarity_sum = torch.sum(similarity_matrix * mask)

    return intra_similarity_sum


def diversity_loss(inputs, targets):
    similarity = 0.0
    inputs = inputs.reshape(inputs.shape[0], -1)
    for i in range(torch.max(targets) + 1):
        temp_inputs = inputs[targets == i]
        similarity += sum_intra_similarity(temp_inputs)
    return similarity/inputs.shape[0].detach()


# 示例输入数据
# 这是一个3x4的张量，表示3个样本，每个样本有4个特征
data = torch.tensor([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]], dtype=torch.float)

# 计算类内余弦相似度
result = sum_intra_similarity(data)

print(result)
