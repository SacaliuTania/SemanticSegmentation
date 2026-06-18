import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as seg_models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

IMAGES_DIR = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/train"
MASKS_DIR  = "D:/facultate/An3/An3_sem2/segmentare_semantica/gtFine_trainvaltest/gtFine/train"
TEST_IMAGE = "D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/bonn/bonn_000045_000019_leftImg8bit.png"
BEST_MODEL = "best_deeplabv3_cityscapes.pth"
OUTPUT_PNG = "segmentare_rezultat.png"

NUM_CLASSES = 19
EPOCHS      = 20
BATCH_SIZE  = 2
LR          = 3e-5
IMG_SIZE    = (256, 512)
MAX_BATCHES = 500

LABEL_MAP = {
    7: 0,  8: 1,  11: 2, 12: 3,  13: 4,
    17: 5, 19: 6, 20: 7, 21: 8,  22: 9,
    23:10, 24:11, 25:12, 26:13,  27:14,
    28:15, 31:16, 32:17, 33:18,
}

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

class CityscapesDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.pairs = []
        for city in sorted(Path(images_dir).iterdir()):
            if not city.is_dir():
                continue
            for img_path in sorted(city.glob("*_leftImg8bit.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                mask_path = Path(masks_dir) / city.name / f"{stem}_gtFine_labelIds.png"
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))
        print(f"  {len(self.pairs)} perechi gasite")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert("RGB").resize(
            (IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        mask = Image.open(mask_path).resize(
            (IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)

        mask_np  = np.array(mask, dtype=np.int64)
        new_mask = np.full_like(mask_np, 255)
        for orig_id, new_id in LABEL_MAP.items():
            new_mask[mask_np == orig_id] = new_id

        pixel_values = TF.normalize(
            TF.to_tensor(img),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return pixel_values, torch.tensor(new_mask, dtype=torch.long)
def build_model():
    # Porneste cu backbone ResNet101 pre-antrenat pe COCO+VOC
    model = seg_models.deeplabv3_resnet101(
        weights=seg_models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=True
    )
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model

def vizualizeaza(model, device, save_path=OUTPUT_PNG):
    if not Path(TEST_IMAGE).exists():
        print(f"Imaginea de test nu exista: {TEST_IMAGE}")
        return

    img = Image.open(TEST_IMAGE).convert("RGB").resize(
        (IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)

    input_tensor = TF.normalize(
        TF.to_tensor(img),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs   = model(input_tensor)
        logits_up = nn.functional.interpolate(
            outputs["out"],
            size=IMG_SIZE,
            mode="bilinear",
            align_corners=False,
        )

    pred_np = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    clase   = np.unique(pred_np)
    print(f"  Clase detectate ({len(clase)}): {[CLASSES[c] for c in clase]}")

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
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Salvat: {save_path}")
    plt.show()

def train():
    if not torch.cuda.is_available():
        print("CUDA nu e disponibil!")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Incarc modelul...")
    model = build_model()
    model.to(device)

    print("Incarc datasetul...")
    dataset = CityscapesDataset(IMAGES_DIR, MASKS_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),       "lr": LR * 0.1},
        {"params": model.classifier.parameters(),     "lr": LR},
        {"params": model.aux_classifier.parameters(), "lr": LR},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7)

    loss_fn   = nn.CrossEntropyLoss(ignore_index=255)
    scaler    = torch.cuda.amp.GradScaler(init_scale=128)
    best_loss = float("inf")

    print(f"Incepe antrenarea ({EPOCHS} epoci, max {MAX_BATCHES} batches/epoca)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss    = 0.0
        valid_batches = 0

        for i, (pixels, labels) in enumerate(loader):
            if i >= MAX_BATCHES:
                break

            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(pixels)

                logits_up = nn.functional.interpolate(
                    outputs["out"], size=labels.shape[-2:],
                    mode="bilinear", align_corners=False)
                loss_main = loss_fn(logits_up, labels)

                aux_up = nn.functional.interpolate(
                    outputs["aux"], size=labels.shape[-2:],
                    mode="bilinear", align_corners=False)
                loss_aux = loss_fn(aux_up, labels)

                loss = loss_main + 0.4 * loss_aux

            if torch.isnan(loss):
                print(f"  NaN la batch {i}, sar peste!")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item()
            valid_batches += 1

            if i % 20 == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  Ep {epoch+1}/{EPOCHS} | Batch {i:3d}/{MAX_BATCHES} "
                      f"| Loss: {loss.item():.4f} | VRAM: {vram:.1f}GB")

        scheduler.step()
        avg = total_loss / max(valid_batches, 1)
        print(f"Epoca {epoch+1} | Loss mediu: {avg:.4f}")

        if avg < best_loss and valid_batches > 0:
            best_loss = avg
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"model salvat (loss={best_loss:.4f})")

    print(f"\nIncarc best model din {BEST_MODEL}...")
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))

    print("Vizualizare pe imaginea de test...")
    vizualizeaza(model, device)


if __name__ == "__main__":
    train()