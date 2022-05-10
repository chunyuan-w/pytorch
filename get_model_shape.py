import torch
import torchvision

# model_name = "resnet50"
model_name = "resnext101_32x8d"
model = getattr(torchvision.models, model_name)().eval()

x = torch.randn(24, 3, 224, 224)

model(x)

print(model)