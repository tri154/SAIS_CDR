import torch

a = torch.rand(40, 512)

att = torch.rand(12, 40, 40)


# res1 = (att.mean(0) / att.mean(0).sum(-1)) @ a
res1 = att.mean(0) @ a
res1 = res1.mean(0)
print(res1[:10])

temp = att.mean(-1).mean(0)
# temp = temp / torch.sum(temp)
res2 = temp @ a
print(res2[:10])


print(torch.sum((res1 - res2)**2))
