import torch
import torchvision

# model_name = "resnet50"
# model_name = "resnext101_32x8d"
# model = getattr(torchvision.models, model_name)().eval()

# model = torchvision.models.mobilenet_v2(inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])

model = torchvision.models.shufflenet_v2_x0_5()

x = torch.randn(24, 3, 224, 224)

model(x)

print(model)