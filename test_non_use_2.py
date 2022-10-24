import torch

t = torch.rand((4, 2))
t = t * 100
print(t)
t_min, t_max = t.min(), t.max()
new_min, new_max = 0, 1
t = (t - t_min)/(t_max - t_min)*(new_max - new_min) + new_min
print(t)

m = torch.nn.LogSoftmax(dim=1)
output = m(t)
loss_domain = torch.nn.NLLLoss()
loss = loss_domain(output[:, 1], torch.tensor([0, 0, 0, 0]))
print(loss)

a = (1, 2)
print(a[0])


nan = torch.tensor(float('nan'))
if torch.isnan(nan):
    print(nan)


exit()
