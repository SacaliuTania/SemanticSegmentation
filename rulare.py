import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as torch_F
import torchvision.models.segmentation as seg_models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IMAGE_PATH = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
OUTPUT_PNG = "segmentare_cityscapes.png"
ALPHA      = 0.55
IMG_SIZE   = (512, 1024)

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

COLORMAP = np.array([
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
], dtype=np.uint8)

WEIGHTS_PATH = "best_deeplabv3_resnet101_cityscapes_os16.pth"

print("[Model] Se construieste DeepLabV3 ResNet101...")

model = seg_models.deeplabv3_resnet101(weights=None, num_classes=19, aux_loss=True)

if os.path.exists(WEIGHTS_PATH):
    print(f"[Model] Se incarca greutati din {WEIGHTS_PATH}...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print("[Model] Incarcat cu succes.")
else:
    print("[Model] ATENTIE: fisierul de greutati nu a fost gasit!")
    print(f"         Descarca best_deeplabv3_resnet101_cityscapes_os16.pth")
    print(f"         de la https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5yfhnjIJAHLdpewa")
    print(f"         si pune-l in: {os.path.abspath('.')}")
    exit(1)

model.eval()

print(f"[Input] {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert("RGB").resize(
    (IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR
)

input_tensor = TF.to_tensor(img)
input_tensor = TF.normalize(
    input_tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
input_tensor = input_tensor.unsqueeze(0)

print("[Inferenta] ")
with torch.no_grad():
    outputs = model(input_tensor)

logits_up = torch_F.interpolate(
    outputs["out"],
    size=IMG_SIZE,
    mode="bilinear",
    align_corners=False
)
pred_np = logits_up.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
clase   = np.unique(pred_np)

print("\n[Clase detectate]")
for cls_id in clase:
    n   = int(np.sum(pred_np == cls_id))
    pct = 100.0 * n / pred_np.size
    print(f"  {cls_id:2d}  {CLASSES[cls_id]:<15s}  {n:>8d} px  ({pct:.2f}%)")

seg_color = COLORMAP[pred_np]
img_arr   = np.array(img, dtype=np.float32)
overlay   = (ALPHA * seg_color.astype(np.float32) + (1 - ALPHA) * img_arr).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.patch.set_facecolor("#111111")

for ax, (imagine, titlu) in zip(axes, [
    (img,                        "Imagine originala"),
    (Image.fromarray(seg_color), "Segmentare semantica (Cityscapes 19 clase)"),
    (Image.fromarray(overlay),   "Overlay"),
]):
    ax.imshow(imagine)
    ax.set_title(titlu, color="white", fontsize=12, pad=8)
    ax.axis("off")

patches = [
    mpatches.Patch(
        facecolor=COLORMAP[cls_id] / 255.0,
        edgecolor="white",
        linewidth=0.5,
        label=f"{cls_id}: {CLASSES[cls_id]}"
    )
    for cls_id in clase
]

fig.legend(
    handles=patches,
    loc="lower center",
    ncol=min(len(clase), 10),
    bbox_to_anchor=(0.5, 0.01),
    fontsize=10,
    framealpha=0.3,
    labelcolor="white",
    facecolor="#222222",
    edgecolor="gray",
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[Salvat] {OUTPUT_PNG}")
plt.show()