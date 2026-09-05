import os
import sys
import torchvision
from torchvision.datasets import GTSRB

def download_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading GTSRB Dataset to {data_dir}...")
    
    try:
        print("Downloading GTSRB train split...")
        train_data = GTSRB(root=data_dir, split='train', download=True)
        print(f"Train split downloaded successfully! Samples count: {len(train_data)}")
        
        print("Downloading GTSRB test split...")
        test_data = GTSRB(root=data_dir, split='test', download=True)
        print(f"Test split downloaded successfully! Samples count: {len(test_data)}")
        
        print("GTSRB dataset ready!")
        return True
    except Exception as e:
        print(f"Error downloading GTSRB via torchvision: {e}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    if not success:
        sys.exit(1)
