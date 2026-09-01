import numpy as np
import torch


def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))


# draw_examples("arange", [{"" : arange(i)} for i in [5, 3, 9]])

a = np.ones((6, 7))
# print(a[:].shape)
# print(a[:, :].shape)
# print(a[:, :, None].shape)
# print(a[None, :].shape)
# print(a[:, None].shape)

print(arange(7)[:, None].shape)
