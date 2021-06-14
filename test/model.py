import torch
import torch.nn as nn
import torch.nn.functional as F

class EaseClassificationModel(nn.Module):

    def __init__(self, H, W):
        super(EaseClassificationModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * (H//4) * (W//4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

if __name__ == '__main__':  
    model = EaseClassificationModel()
    img = torch.randn(1, 3, 64, 64).float()
    output = model(img)
    print(output)