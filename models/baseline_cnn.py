# Step3
'''
Define the CNN architecture (Baseline) 
'''

import torch
import torch.nn as nn

class DefectBaselineCNN(nn.Module):


    def __init__(self, num_classes: int=6, dropout_p: float = 0.4):
        super(DefectBaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels | 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Slides learned filters over the image patches to extract local structural features.
            nn.BatchNorm2d(32), # Normalizes layer activations, stabilizing gradient updates and speeding up training.
            nn.ReLU(inplace=True), # Introduces non-linearity so the network can learn complex boundary curves.
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsamples spatial resolution by half, reducing computational cost while keeping prominent activations.


            # Block 2: 32 -> 64 channels | 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels | 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 -> 256 channels | 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling ensures consistent output size regardless of input spatial
        self.pool = nn.AdaptiveAvgPool2d((4,4))

        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens the 3D feature tensor into a 1D vector of length $4096$ ($256 \times 4 \times 4$).
            nn.Linear(256 * 4 * 4, 256), # Dense layer that learns combinations of the extracted defect features.
            nn.ReLU(inplace=True), 
            nn.Dropout(p=dropout_p), # randomly deactivates 40% of neurons during training, preventing the model from memorizing specific training samples (regularization against overfitting).
            nn.Linear(256, num_classes) # Maps the final 256 hidden features to 6 raw logits (one score for each defect class: open, short, mousebite, spur, copper, pin-hole).
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    model = DefectBaselineCNN(num_classes=6)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Baseline CNN successfully initialized.")
    print(f"Output shape: {output.shape} (Expected: [2, 6])")