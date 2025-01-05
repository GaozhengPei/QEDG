from __future__ import print_function
import argparse  # Python 命令行解析工具
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import os
from nets import VGG19,resnet18


def test_acc(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    return acc, test_loss


def get_dataset(dataset):
    data_dir = './data/{}'.format(dataset)
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]),
                                       download=True)
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]), download=True)
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()])
                                              , download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ])
                                             , download=True)
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]),
                                      download=True)
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]), download=True)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]),
                                         download=True)
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]),
                                        download=True)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,download=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))
    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
        # train_loader = data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=4)
        # val_loader = data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=4)

    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=True, num_workers=0)

    return train_loader, test_loader


def train(model, train_loader, optimizer):
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='dataset')
    parser.add_argument('--net', type=str, default="resnet18",
                        help='dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args.dataset)
    model = resnet18(args.dataset).cuda()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    bst_acc = -1
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer)
        acc, loss = test_acc(model, test_loader)
        if acc > bst_acc:
            bst_acc = acc
            torch.save(model.state_dict(), './target_model_weight/{}_{}.pth'.format(args.net, args.dataset))

        bst_acc = max(bst_acc, acc)
        print("Epoch:{},\t test_acc:{}, best_acc:{}".format(epoch, acc, bst_acc))


if __name__ == '__main__':
    main()
