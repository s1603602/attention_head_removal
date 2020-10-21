import random
import torch

def remove_head(p, mha, istrain):
    if not istrain:
        return mha

    batch, head, time, d_k = mha.shape

    # dropout probability is p
    if random.random() < p:
        mask = torch.ones([batch, 1, time, d_k], dtype=torch.uint8).cuda()  # masked_fill fill ones
    else:
        mask = torch.zeros([batch, 1, time, d_k], dtype=torch.uint8).cuda()

    for _ in range(head - 1):
        if random.random() < p:
            maskH = torch.ones([batch, 1, time, d_k], dtype=torch.uint8).cuda()  # masked_fill fill ones
        else:
            maskH = torch.zeros([batch, 1, time, d_k], dtype=torch.uint8).cuda()
        mask = torch.cat((mask, maskH), dim=1)

    mha = mha.masked_fill(mask, 0)
    return mha * (1 / (1 - p))

