"""
train_vn.py  –  Train TrafficSignCNN from scratch on 15 Vietnamese sign classes.
Reads images from data/vn_signs/<class_id>/*.png
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'vn_signs')
MODEL_OUT  = os.path.join(BASE_DIR, 'best_model_vn.pth')
MAP_OUT    = os.path.join(BASE_DIR, 'class_mapping_vn.json')

NUM_CLASSES = 15
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
VAL_RATIO   = 0.15
SEED        = 42

# ── Class Metadata ─────────────────────────────────────────────────────────────
CLASS_META = [
    {"id":0,  "code":"W.201a", "name_vi":"Chỗ ngoặt nguy hiểm vòng bên trái",  "name_en":"Dangerous curve to the left",    "category":"Warning"},
    {"id":1,  "code":"W.201b", "name_vi":"Chỗ ngoặt nguy hiểm vòng bên phải",  "name_en":"Dangerous curve to the right",   "category":"Warning"},
    {"id":2,  "code":"W.207a", "name_vi":"Đường hai chiều",                     "name_en":"Two-way traffic",                "category":"Warning"},
    {"id":3,  "code":"W.208",  "name_vi":"Giao nhau với đường sắt có rào chắn", "name_en":"Level crossing with barrier",    "category":"Warning"},
    {"id":4,  "code":"W.224",  "name_vi":"Đường trơn trượt",                    "name_en":"Slippery road",                  "category":"Warning"},
    {"id":5,  "code":"W.245",  "name_vi":"Trẻ em",                              "name_en":"Children crossing",              "category":"Warning"},
    {"id":6,  "code":"P.102",  "name_vi":"Đường cấm",                           "name_en":"No entry for all vehicles",      "category":"Prohibitory"},
    {"id":7,  "code":"P.103a", "name_vi":"Cấm xe ô tô",                         "name_en":"No automobiles",                 "category":"Prohibitory"},
    {"id":8,  "code":"P.104",  "name_vi":"Cấm xe mô tô",                        "name_en":"No motorcycles",                 "category":"Prohibitory"},
    {"id":9,  "code":"P.123",  "name_vi":"Cấm vượt",                            "name_en":"No overtaking",                  "category":"Prohibitory"},
    {"id":10, "code":"P.127-40","name_vi":"Tốc độ tối đa 40 km/h",             "name_en":"Speed limit (40km/h)",           "category":"Prohibitory"},
    {"id":11, "code":"P.127-60","name_vi":"Tốc độ tối đa 60 km/h",             "name_en":"Speed limit (60km/h)",           "category":"Prohibitory"},
    {"id":12, "code":"R.301a", "name_vi":"Hướng đi phải theo: Đi thẳng",       "name_en":"Ahead only",                     "category":"Mandatory"},
    {"id":13, "code":"R.302a", "name_vi":"Hướng đi phải theo: Rẽ phải",        "name_en":"Turn right ahead",               "category":"Mandatory"},
    {"id":14, "code":"R.303",  "name_vi":"Giao thông hai chiều",                "name_en":"Two-way traffic (mandatory)",    "category":"Mandatory"},
]

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class VNSignDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        for cls_id in range(NUM_CLASSES):
            cls_dir = os.path.join(data_dir, str(cls_id))
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith('.png'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_id))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

# ── Model (same arch, new num_classes) ────────────────────────────────────────
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=15, input_shape=(3,32,32)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,*input_shape)
            flat = self.features(dummy).view(1,-1).size(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat,256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# ── Training ───────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all data (no transform yet for split)
    full_ds = VNSignDataset(DATA_DIR)
    n_val   = int(len(full_ds) * VAL_RATIO)
    n_train = len(full_ds) - n_val
    train_idx, val_idx = random_split(range(len(full_ds)), [n_train, n_val],
                                      generator=torch.Generator().manual_seed(SEED))

    class SubsetWithTransform(Dataset):
        def __init__(self, base, indices, transform):
            self.base=base; self.indices=list(indices); self.transform=transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            path,label = self.base.samples[self.indices[i]]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label

    train_ds = SubsetWithTransform(full_ds, train_idx.indices, TRAIN_TRANSFORM)
    val_ds   = SubsetWithTransform(full_ds, val_idx.indices,   VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = TrafficSignCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # Train
        model.train(); train_loss=0; train_correct=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs); loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()*imgs.size(0)
            train_correct += (out.argmax(1)==labels).sum().item()
        scheduler.step()

        # Validate
        model.eval(); val_correct=0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1)==labels).sum().item()

        train_acc = train_correct/len(train_ds)*100
        val_acc   = val_correct/len(val_ds)*100
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss/len(train_ds):.4f}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"           ^ New best val_acc={val_acc:.1f}% — saved to {MODEL_OUT}")

    print(f"\nTraining complete. Best val_acc: {best_acc:.1f}%")

    # Export class_mapping_vn.json
    mapping = {"classes": CLASS_META, "model_metadata": {
        "model_name": "TrafficSignCNN_VN",
        "framework": "PyTorch -> ONNX",
        "input_name": "input", "output_name": "output",
        "input_shape": [1,3,32,32], "channel_order": "RGB",
        "normalization": {"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        "num_classes": NUM_CLASSES
    }}
    with open(MAP_OUT,'w',encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"class_mapping_vn.json saved → {MAP_OUT}")

if __name__ == '__main__':
    main()

