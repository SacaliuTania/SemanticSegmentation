import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())

model = deeplabv3_resnet50(pretrained=True)
model.eval()

x = torch.randn(1, 3, 512, 512)
out = model(x)

print("Output shape:", out['out'].shape)