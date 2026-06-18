import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import cv2

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()

img = cv2.imread("D:\\facultate\\An3\\An3_sem2\\segmentare_semantica\\leftImg8bit_trainvaltest\\leftImg8bit\\val\\frankfurt\\frankfurt_000000_000294_leftImg8bit.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img_rgb/255.0, dtype=torch.float).permute(2,0,1).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)['out'][0]

mask = output.argmax(0).byte().numpy()
cv2.imshow("Mask", mask*10)  # *10 doar pentru vizualizare culori
cv2.waitKey(0)