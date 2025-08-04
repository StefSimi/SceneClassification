import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_deeplabv3_model(num_classes=9, pretrained=True):
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = deeplabv3_resnet50(weights=weights, aux_loss=True)  #
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  #
    return model


if __name__ == '__main__':
    model = get_deeplabv3_model()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(out['out'].shape)
