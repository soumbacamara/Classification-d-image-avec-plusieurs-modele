'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Branche shortcut (connexion résiduelle)
        if stride == 1 and in_planes == planes:
            self.shortcut = nn.Identity()  # Fonction identité si pas de changement de taille
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        out = F.relu(self.bn1(self.conv1(x)))
        # 2. Go through conv2, bn
        out = self.bn2(self.conv2(out))
        # 3. Combine with shortcut output, and go through relu
        out += self.shortcut(x)
        # 4) Activation finale
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        """
        1) conv1 + bn1 + relu
        2) layer1 -> layer2 -> layer3 -> layer4
        3) average pooling global
        4) fully-connected (linear) vers num_classes
        """
        # 1) Convolution initiale
        x = self.conv1(images)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)
        # 2) Passage dans chaque "layer"
        x = self.layer1(x)  # [N, 64, 32, 32] si entrée 32x32
        x = self.layer2(x)  # [N, 128, 16, 16]
        x = self.layer3(x)  # [N, 256, 8, 8]
        x = self.layer4(x)  # [N, 512, 4, 4]
        
        # 3) Average Pooling global (ici, pool size=4 pour CIFAR-10)
        # Si vous utilisez un autre dataset, adaptez la taille
        # x = F.avg_pool2d(x, 4)  # [N, 512, 1, 1]
        x = self.avg_pool(x)  # [N, 512, 1, 1]

        # 4) Aplatir et passer dans la couche fully-connected
        x = x.view(x.size(0), -1)  # [N, 512]
        logits = self.linear(x)        # [N, num_classes]
        return logits
    def visualize(self, logdir):
        """ Visualize the kernel in the desired directory """
        raise NotImplementedError
