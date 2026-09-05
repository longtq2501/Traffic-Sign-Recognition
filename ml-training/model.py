import torch
import torch.nn as nn

class TrafficSignCNN(nn.Module):
    """
    Convolutional Neural Network for Traffic Sign Recognition (GTSRB 43 classes).
    Uses NCHW input format [batch, channels, height, width] -> [1, 3, 32, 32].
    Computes linear classifier input dimension dynamically.
    """
    def __init__(self, num_classes: int = 43, input_shape: tuple = (3, 32, 32)):
        super(TrafficSignCNN, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Feature extractor: 2 Conv Blocks with BatchNorm, ReLU, MaxPool & Dropout
        self.features = nn.Sequential(
            # Block 1: (3, 32, 32) -> (64, 16, 16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Calculate dynamic flatten feature dimension
        flatten_size = self._get_flatten_size(input_shape)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_size(self, input_shape: tuple) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.
        Args:
            x (Tensor): Input tensor with shape [batch_size, 3, 32, 32] (NCHW, RGB).
        Returns:
            Tensor: Raw logits with shape [batch_size, num_classes].
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = TrafficSignCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"TrafficSignCNN initialized successfully.")
    print(f"  Input shape : {dummy_input.shape} (NCHW format)")
    print(f"  Output shape: {output.shape} (43 logits)")
