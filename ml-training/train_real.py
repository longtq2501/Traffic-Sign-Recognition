"""
train_real.py
Train TrafficSignCNN using real user-supplied images from dataset_real/.
Features:
- Reads images from dataset_real/<class_folder>/*.{png,jpg,jpeg,webp}
- If a folder has real images, uses aggressive augmentation to expand to 100+ training samples.
- If a folder is empty, gracefully falls back to synthetic data in data/vn_signs/<class_id>.
- Exports directly to ONNX and copies to backend resources automatically.
"""

import os, glob, random, shutil, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import onnx
import onnxruntime as ort

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR   = os.path.join(BASE_DIR, 'dataset_real')
SYNTH_DATA_DIR  = os.path.join(BASE_DIR, 'data', 'vn_signs')
MODEL_PTH       = os.path.join(BASE_DIR, 'best_model_vn.pth')
MODEL_ONNX      = os.path.join(BASE_DIR, 'traffic_sign_model.onnx')
BACKEND_RES_ONNX = os.path.join(BASE_DIR, '..', 'backend', 'src', 'main', 'resources', 'model', 'traffic_sign_model.onnx')
BACKEND_TGT_ONNX = os.path.join(BASE_DIR, '..', 'backend', 'target', 'classes', 'model', 'traffic_sign_model.onnx')

NUM_CLASSES = 15
EPOCHS      = 25
BATCH_SIZE  = 32
LR          = 1e-3

# ── CNN Architecture ──────────────────────────────────────────────────────────
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

# ── Augmentation Helper ────────────────────────────────────────────────────────
def augment_image(img):
    # Random slight rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    # Random crop / zoom
    w, h = img.size
    scale = random.uniform(0.85, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    left = random.randint(0, w - nw)
    top = random.randint(0, h - nh)
    img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

    # Lighting variations
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.25))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))

    return img

class RealSignDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (PIL.Image, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data():
    samples_train = []
    samples_val = []

    transform_norm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    subdirs = sorted(os.listdir(REAL_DATA_DIR))
    print(f"\nScanning real images in {REAL_DATA_DIR}...")

    stats = []

    for cid in range(NUM_CLASSES):
        # Match folder starting with cid or f"{cid:02d}"
        prefix = f"{cid:02d}_"
        prefix_alt = f"{cid}_"
        target_dir = None
        for d in subdirs:
            if d.startswith(prefix) or d.startswith(prefix_alt) or d == str(cid):
                target_dir = os.path.join(REAL_DATA_DIR, d)
                break

        img_paths = []
        if target_dir and os.path.exists(target_dir):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp'):
                img_paths.extend(glob.glob(os.path.join(target_dir, ext)))

        real_count = len(img_paths)
        class_images = []

        if real_count > 0:
            # User provided real images
            for p in img_paths:
                try:
                    im = Image.open(p).convert('RGB')
                    class_images.append(im)
                except Exception as e:
                    print(f"  Warning: could not open {p}: {e}")

            # Expand with augmentations up to at least 150 images
            augmented = []
            multiplier = max(1, 150 // len(class_images))
            for im in class_images:
                augmented.append(im)  # original
                for _ in range(multiplier):
                    augmented.append(augment_image(im))

            # Split 85% train / 15% val
            random.shuffle(augmented)
            split = int(len(augmented) * 0.85)
            for im in augmented[:split]:
                samples_train.append((im, cid))
            for im in augmented[split:]:
                samples_val.append((im, cid))

            stats.append(f"  [Class {cid:02d}] REAL: {real_count} raw images -> expanded to {len(augmented)} samples")
        else:
            # Fallback to synthetic data if available
            synth_dir = os.path.join(SYNTH_DATA_DIR, str(cid))
            synth_paths = glob.glob(os.path.join(synth_dir, '*.png')) if os.path.exists(synth_dir) else []
            for p in synth_paths:
                try:
                    im = Image.open(p).convert('RGB')
                    class_images.append(im)
                except:
                    pass

            if class_images:
                random.shuffle(class_images)
                split = int(len(class_images) * 0.85)
                for im in class_images[:split]:
                    samples_train.append((im, cid))
                for im in class_images[split:]:
                    samples_val.append((im, cid))
                stats.append(f"  [Class {cid:02d}] SYNTHETIC fallback: {len(class_images)} samples")
            else:
                stats.append(f"  [Class {cid:02d}] EMPTY (no data)")

    for s in stats:
        print(s)

    print(f"\nTotal Dataset: {len(samples_train)} train, {len(samples_val)} val")
    return (
        RealSignDataset(samples_train, transform_norm),
        RealSignDataset(samples_val, transform_norm)
    )

def train_and_export():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds, val_ds = load_data()
    if len(train_ds) == 0:
        print("Error: No training data found!")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device} for {EPOCHS} epochs...")

    model = TrafficSignCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
        scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()

        t_acc = (train_correct / len(train_ds)) * 100.0
        v_acc = (val_correct / len(val_ds)) * 100.0
        print(f"Epoch {epoch:2d}/{EPOCHS} - train_acc: {t_acc:.1f}%, val_acc: {v_acc:.1f}%")

        if v_acc >= best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_PTH)

    print(f"\nTraining complete! Best Validation Accuracy: {best_acc:.1f}%")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print("\nExporting model to ONNX format...")
    model.load_state_dict(torch.load(MODEL_PTH, map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy,
        MODEL_ONNX,
        export_params=True,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )

    onnx_model = onnx.load(MODEL_ONNX)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {MODEL_ONNX}")

    # Copy to backend resources
    for dest in [BACKEND_RES_ONNX, BACKEND_TGT_ONNX]:
        if os.path.exists(os.path.dirname(dest)):
            shutil.copy2(MODEL_ONNX, dest)
            print(f"Deployed to backend: {dest}")

    print("\nAll done! You can now restart your backend server to load the new weights.")

if __name__ == '__main__':
    train_and_export()
