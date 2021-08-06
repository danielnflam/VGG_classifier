import torchvision.models
import torch
import torch.nn as nn

class VGG16_pretrained(nn.Module):
    def __init__(self, N_CLASSES, verbose):
        super().__init__()
        self.N_CLASSES = N_CLASSES
        self.verbose = verbose
        
        vgg16_pt = torchvision.models.vgg16(pretrained=True)
        num_in_ftrs = vgg16_pt.classifier[6].in_features
        # Set output classes to N_CLASSES
        vgg16_pt.classifier[6] = nn.Linear(num_in_ftrs, self.N_CLASSES)
        
        if self.verbose:
            print(vgg16_pt)
            total_num = sum(p.numel() for p in vgg16_pt.parameters())
            trainable_num = sum(p.numel() for p in vgg16_pt.parameters() if p.requires_grad)
            print('Parameters total: ',total_num)
            print('Parameters trainable: ',trainable_num)
        self.pretrained = vgg16_pt
    def forward(self,x):
        x = self.pretrained(x)
        return x

class VGG16_Rajaraman(nn.Module):
    def __init__(self, N_CLASSES, max_layer_to_freeze=0, verbose=False):
        super().__init__()
        self.N_CLASSES=N_CLASSES
        self.verbose = verbose
        self.max_layer_to_freeze = max_layer_to_freeze
        
        # Pretrained
        vgg16_pt = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv = vgg16_pt.features
        # Freeze layers
        for layer_idx in range(self.max_layer_to_freeze):
            for param in self.vgg16_conv[layer_idx].parameters():
                param.requires_grad = False
        
        self.GAP = nn.AdaptiveAvgPool2d((1,1)) # global average pooling
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, self.N_CLASSES)
        
        
        model = nn.Sequential(self.vgg16_conv, self.GAP, self.classifier)
        if verbose:
            print(model)
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Parameters total: ',total_num)
        print('Parameters trainable: ',trainable_num)
    
    def forward(self, x):
        x = self.vgg16_conv(x)
        x = self.GAP(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x