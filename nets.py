import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

nz = 128
nc = 3


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, activation=None, final_bn=True):
        super(GeneratorA, self).__init__()

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False)
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False)
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x:
            return img
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            return self.activation(img)


class GeneratorC(nn.Module):
    '''
    Conditional Generator
    '''

    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(GeneratorC, self).__init__()

        self.label_emb = nn.Embedding(num_classes, nz)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz * 2, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """

    def __init__(self, nz=256, ngf=64, nc=3, img_size=32, slope=0.2):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class Generator_2(nn.Module):
    def __init__(self, nz, ngf, img_size, nc):
        super(Generator_2, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


#
# class CNNCifar10(nn.Module):
#     def __init__(self):
#         super(CNNCifar10, self).__init__()
#         self.conv1 = nn.Conv2d(3, 256, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(256, 256, 3)
#         self.conv3 = nn.Conv2d(256, 128, 3)
#         self.fc1 = nn.Linear(128 * 4 * 4, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 128 * 4 * 4)
#         x = self.fc1(x)
#         return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(dataset):
    resnet_18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
    if dataset == 'mnist' or dataset == 'fmnist':
        resnet_18.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                    bias=False)
    if dataset == 'cifar100':
        resnet_18.fc = nn.Linear(in_features=512, out_features=100)
    elif dataset == 'tiny-imagenet':
        resnet_18.fc = nn.Linear(in_features=512, out_features=200)
    else:
        resnet_18.fc = nn.Linear(in_features=512, out_features=10)
    return resnet_18


def resnet34(dataset):
    resnet_34 = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights)
    if dataset == 'mnist' or dataset == 'fmnist':
        resnet_34.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                    bias=False)
    if dataset == 'cifar100':
        resnet_34.fc = nn.Linear(in_features=512, out_features=100)
    elif dataset == 'tiny-imagenet':
        resnet_34.fc = nn.Linear(in_features=512, out_features=200)
    else:
        resnet_34.fc = nn.Linear(in_features=512, out_features=10)
    return resnet_34


def VGG19(dataset):
    vgg_19 = torchvision.models.vgg19()
    if dataset == 'mnist' or dataset == 'fmnist':
        vgg_19.features[0] = nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                 bias=False)
    if dataset == 'cifar100':
        vgg_19.classifier[6] = nn.Linear(in_features=4096, out_features=100)
    elif dataset == 'tiny-imagenet':
        vgg_19.classifier[6] = nn.Linear(in_features=4096, out_features=200)
    else:
        vgg_19.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    return vgg_19


class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_m(nn.Module):
    def __init__(self):
        self.number = 0
        super(Net_m, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2 * 2 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2 * 2 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
