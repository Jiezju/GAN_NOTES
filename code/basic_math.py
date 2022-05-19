import torch
import torch.nn.functional as F

# kl divergence
# kl(p || q) = \sum p log p/q
p = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
q = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float32)

kl = F.kl_div(p, q, size_average=None, reduce=None, reduction='mean')
print(kl)


# JS 散度 nature of gan
# JS = 1 / 2 * (KL(p || (p + q) / 2) + KL(q || (p + q) / 2))
# p, q distribution
def js_div(p, q):
    kl = F.kl_div
    return 0.5 * (kl(p, (p + q) / 2) + kl(q, (p + q) / 2))


# f divergence
def f_div(p, q, func):
    return torch.sum(p * func(p / q))
