import torch.nn as nn
import torch.nn.functional as F

class FCNonsense(nn.Module):
    def __init__(self):
        super(FCNonsense, self).__init__()
        # assuming input x is (1, 240, 320)
        self.fc = nn.Linear(76800, 2)

    def forward(self, x):
        x = x.view(-1)
        x = self.fc(x)
        return x
