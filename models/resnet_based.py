import torch.nn as nn
import torchvision
from torchvision.models import resnet18
import torch.optim as optim

def ResnetBased(n_classes=2):
    # retraining the full net
    # model_ft = resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, n_classes)
    # return model_ft

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # https://discuss.pytorch.org/t/why-torchvision-models-can-not-forward-the-input-which-has-size-of-larger-than-430-430/2067/9
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    # Parameters of newly constructed modules have requires_grad=True by default
    # Only parameters of final layer are being optimized as opoosed to before.
    num_ftrs = model.fc.in_features
    print('resnet fc input feature count:', num_ftrs)
    model.fc = nn.Linear(num_ftrs, n_classes)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return model
