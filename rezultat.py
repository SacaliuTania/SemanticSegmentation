import os

from rulare_train import TEST_IMAGE

# Forțăm Windows-ul să ignore duplicarea OpenMP înainte de absolut orice import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#TEST_IMAGE = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
#TEST_IMAGE = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/bonn/bonn_000045_000019_leftImg8bit.png"
#TEST_IMAGE = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/munich/munich_000038_000019_leftImg8bit.png"
TEST_IMAGE = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/munich/munich_000069_000019_leftImg8bit.png"

BEST_MODEL = "best_deeplabv3_cityscapes.pth"
OUTPUT_PNG = "segmentare_rezultat.png"

NUM_CLASSES = 19
IMG_SIZE    = (256, 512)

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

COLORMAP = np.array([
    [128, 64,128],[244, 35,232],[ 70, 70, 70],[102,102,156],[190,153,153],
    [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],
    [ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],
    [  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]
], dtype=np.uint8)


def build_model():

    model = seg_models.deeplabv3_resnet101(weights=None, aux_loss=True)
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


def ruleaza_predictie_vizuala():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Folosesc dispozitivul: {device}")

    if not Path(TEST_IMAGE).exists():
        print(f"Nu exista imaginea la:\n{TEST_IMAGE}")
        return

    if not Path(BEST_MODEL).exists():
        print(f"Nu exista ponderile: {BEST_MODEL}")
        return
    model = build_model()

    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    model.to(device)
    model.eval()

    img = Image.open(TEST_IMAGE).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR) #procesarea imaginilor de test

    input_tensor = TF.normalize(
        TF.to_tensor(img),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).unsqueeze(0).to(device)

    print("Inferenta")
    with torch.no_grad():
        outputs = model(input_tensor)
        logits_up = nn.functional.interpolate(
            outputs["out"],
            size=IMG_SIZE,
            mode="bilinear",
            align_corners=False,
        )

    pred_np = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    clase   = np.unique(pred_np)
    print(f"   Clase detectate în imagine ({len(clase)}): {[CLASSES[c] for c in clase]}")

    seg_color = COLORMAP[pred_np]
    img_arr   = np.array(img, dtype=np.float32)
    overlay   = (0.55 * seg_color.astype(np.float32) + 0.45 * img_arr).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor("#111111")

    for ax, (imagine, titlu) in zip(axes, [
        (img,                        "Imagine originala"),
        (Image.fromarray(seg_color), "Segmentare semantica"),
        (Image.fromarray(overlay),   "Overlay"),
    ]):
        ax.imshow(imagine)
        ax.set_title(titlu, color="white", fontsize=12)
        ax.axis("off")

    patches = [
        mpatches.Patch(facecolor=COLORMAP[c]/255.0, edgecolor="white",
                       linewidth=0.5, label=f"{c}: {CLASSES[c]}")
        for c in clase
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(clase), 10),
               bbox_to_anchor=(0.5, 0.01), fontsize=10, framealpha=0.3,
               labelcolor="white", facecolor="#222222", edgecolor="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Succes! Rezultatul a fost salvat în: {OUTPUT_PNG}")
    plt.show()

if __name__ == "__main__":
    ruleaza_predictie_vizuala()