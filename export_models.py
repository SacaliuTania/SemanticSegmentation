import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models

NUM_CLASSES = 19
IMG_SIZE = (256, 512)
BEST_MODEL_PATH = "best_deeplabv3_cityscapes.pth"
OUTPUT_LIBTORCH_PATH = "deeplabv3_antrenat.pt"

model = seg_models.deeplabv3_resnet101(weights=None, aux_loss=True)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))

model.eval()

class MyModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        return self.model(x)["out"]

wrapped_model = MyModel(model)
wrapped_model.eval()

example = torch.rand(1, 3, IMG_SIZE[0], IMG_SIZE[1])

converted_model = torch.jit.trace(wrapped_model, example)
converted_model.save(OUTPUT_LIBTORCH_PATH)

print("Model exportat!")