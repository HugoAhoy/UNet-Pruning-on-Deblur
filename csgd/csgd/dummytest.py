import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, 3, 1, 1, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x

model = MyModel()
x = torch.randn(1, 3, 3, 3)
output = model(x)
print(output.shape)
output.mean().backward()
model.zero_grad()
print(model.conv.weight.grad.shape)

# prune
with torch.no_grad():
    model.conv.weight = nn.Parameter(model.conv.weight[:5])

output = model(x)
print(output.shape)
output.mean().backward()
print(model.conv.weight.grad.shape)
model.zero_grad()

# for k, v in model.named_parameters():
#     v.data = v[:5]

# print(model.conv.weight.grad.shape)