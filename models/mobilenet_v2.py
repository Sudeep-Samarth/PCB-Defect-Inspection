import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_mobilenet_v2(num_classes: int = 6, freeze_features: bool = True):
    """
    Builds MobileNetV2 pretrained on ImageNet and adapts its classifier head.
    
    Args:
        num_classes: Number of defect categories (6 for DeepPCB).
        freeze_features: If True, freezes the convolutional feature extractor.
    """
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # MobileNetV2 classifier head: Dropout(p=0.2) -> Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    # Smoke test model instantiation and forward pass
    model = get_mobilenet_v2(num_classes=6, freeze_features=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("MobileNetV2 initialized successfully.")
    print(f"Output shape: {output.shape} (Expected: [2, 6])")