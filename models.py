import torchvision.models
import torch

def get_model_vgg16(N_CLASSES, verbose=False):
    vgg16_pt = torchvision.models.vgg16(pretrained=True)
    num_in_ftrs = vgg16_pt.classifier[6].in_features
    # Set output classes to N_CLASSES
    vgg16_pt.classifier[6] = torch.nn.Linear(num_in_ftrs, N_CLASSES)
    if verbose:
        print(vgg16_pt)
    return vgg16_pt