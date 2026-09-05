import os
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Subset
# pyrefly: ignore [missing-import]
from torchvision import transforms
# pyrefly: ignore [missing-import]
from torchvision.datasets import GTSRB

# Standard Preprocessing Specifications (referenced by Python & Java)
INPUT_SIZE = (32, 32)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 43

# Preprocessing & Data Augmentation Pipelines
# NCHW format: [Channel, Height, Width] -> [3, 32, 32]
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

class TransformDataset(torch.utils.data.Dataset):
    """Wrapper to apply specific transform to a PyTorch dataset subset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(data_dir=None, batch_size=64, num_workers=0, val_ratio=0.15):
    """
    Creates stratified Train, Validation, and Test DataLoaders for GTSRB dataset.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Load raw datasets without transforms first to get targets for stratified split
    raw_train = GTSRB(root=data_dir, split='train', download=False)
    raw_test = GTSRB(root=data_dir, split='test', download=False)
    
    # Extract labels safely for stratified train/val split
    if hasattr(raw_train, '_samples'):
        labels = [y for _, y in raw_train._samples]
    else:
        labels = [raw_train[i][1] for i in range(len(raw_train))]
        
    indices = np.arange(len(labels))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=labels,
        random_state=42
    )
    
    # Create subset datasets with proper transforms
    train_subset = TransformDataset(Subset(raw_train, train_idx), transform=TRAIN_TRANSFORM)
    val_subset = TransformDataset(Subset(raw_train, val_idx), transform=VAL_TEST_TRANSFORM)
    test_dataset = TransformDataset(raw_test, transform=VAL_TEST_TRANSFORM)
    
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    
    print(f"DataLoaders initialized:")
    print(f"  - Train samples: {len(train_subset)} (Stratified)")
    print(f"  - Val samples  : {len(val_subset)} (Stratified)")
    print(f"  - Test samples : {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    t_loader, v_loader, test_loader = get_dataloaders()
    sample_x, sample_y = next(iter(t_loader))
    print(f"Sample batch tensor shape: {sample_x.shape} (NCHW format)")
    print(f"Sample batch labels shape: {sample_y.shape}")
