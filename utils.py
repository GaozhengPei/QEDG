import cv2
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack, GradientSignAttack
import os
import random
from torchvision import datasets, transforms
from nets import resnet18, resnet34, VGG19
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms
from kornia import augmentation


def get_target_model(dataset, model_name):
    pretraind = 'target_model_weight/'
    weight_name = model_name + '_' + dataset + '.pth'
    weight_path = pretraind + weight_name
    state_dict = torch.load(weight_path)
    if model_name == 'resnet34':
        model = resnet34(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'VGG19':
        model = VGG19(dataset).cuda()
        model.load_state_dict(state_dict)
    if model_name == 'resnet18':
        model = resnet18(dataset).cuda()
        model.load_state_dict(state_dict)
    return model


def get_substitute_model(dataset, model_name):
    if model_name == 'resnet34':
        model = resnet34(dataset).cuda()
    if model_name == 'vgg19':
        model = VGG19(dataset).cuda()
    if model_name == 'resnet18':
        model = resnet18(dataset).cuda()
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


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


def test_robust(sub_net, tar_net, test_loader, attack='FGSM', target=False):
    sub_net.eval()
    tar_net.eval()
    adversary = GradientSignAttack(sub_net,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=32/255,targeted=target)
    correct = 0
    total = 0
    for each in test_loader:
        images = each[0].cuda()
        labels = each[1].cuda()
        idx = torch.argmax(tar_net(images), dim=1) == labels
        images = images[idx]
        labels = labels[idx]
        total += len(labels)
        adv_images = adversary.perturb(images, labels)
        predict = torch.argmax(tar_net(adv_images), dim=1)
        correct += (predict != labels).sum()
    return correct / total * 100.

def write(acc, con, best_con, asr, dataset):
    with open('./{}.txt'.format(dataset), 'a') as f:
        f.write("Accuracy of the substitute model:{:.3} %\n".format(acc))
        print("Accuracy of the substitute model:{:.3} %".format(acc))
        f.write("Consistency of substitute model and black box model:{:.3}%, best Consistency:{:.3} % \n".format(con,
                                                                                                                 best_con))
        print("Consistency of substitute model and black box model:{:.3}%, best Consistency:{:.3} %".format(con,
                                                                                                            best_con))
        f.write("Attack success rate:{:.3} %\n".format(asr))
        print("Attack success rate:{:.3} %".format(asr))


def test_cohen_kappa(s_net, t_net, test_loader):
    s_pred = []
    t_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            s_out = torch.softmax(s_net(data), dim=1)
            t_out = torch.softmax(t_net(data), dim=1)
            s_pred += torch.argmax(s_out, dim=1).tolist()
            t_pred += torch.argmax(t_out, dim=1).tolist()
    return cohen_kappa_score(s_pred, t_pred) * 100


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
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
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


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def save_image_batch(imgs, root, _idx, transform, blackBox_net, student, dataset, threshold):
    for idx in range(imgs.shape[0]):
        img = imgs[idx]
        if dataset == 'mnist' or dataset == 'fmnist':
            img = (transform(img).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')[0]
            # img = (transform(img)[0].permute(1,2,0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        else:
            img = (transform(img).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype('uint8')
        img_path = os.path.join(root, dataset, 'images')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_path = os.path.join(img_path, str(_idx) + '-{}.png'.format(idx))
        cv2.imwrite(img_path, img)
        if dataset == 'mnist' or dataset == 'fmnist':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            temp_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float() / 255
        else:
            temp_img = torch.from_numpy(cv2.imread(img_path)).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
        s_out = torch.max(torch.softmax(student(temp_img), dim=1), dim=1)[0]
        if s_out <= threshold:
            with open(os.path.join(root, dataset, 'images_list.txt'), 'a') as f:
                f.write(str(_idx) + '-{}.png'.format(idx) + '\n')
            label = torch.argmax(blackBox_net(temp_img), dim=1)
            with open(os.path.join(root, dataset, 'labels_list.txt'), 'a') as f:
                f.write(str(int(label)))


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset):
        self.root = os.path.abspath(root)
        self.dataset = dataset
        with open(os.path.join(root, dataset, 'images_list.txt'), 'r') as f:
            self.images_names = f.readlines()
        with open(os.path.join(root, dataset, 'labels_list.txt'), 'r') as f:
            self.labels = f.read()

    def __getitem__(self, idx):
        image_name = self.images_names[idx].strip()
        image_path = os.path.join(self.root, self.dataset, 'images', image_name)
        if self.dataset == 'mnist' or self.dataset == 'fmnist':
            img = torch.from_numpy(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0) / 255
        else:
            img = torch.from_numpy(cv2.imread(image_path)).permute(2, 0, 1) / 255
        label = torch.tensor(int(self.labels[idx]))
        return img, label

    def __len__(self):
        return len(self.images_names)


class ImagePool(object):
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset
        self._idx = 0

    def add(self, imgs, transform, blackBox_net, student, threshold):
        save_image_batch(imgs, self.root, self._idx, transform, blackBox_net, student, self.dataset, threshold)
        self._idx += 1

    def get_dataset(self):
        return UnlabeledImageDataset(self.root, self.dataset)


def hard_label_to_one_hot(hard_label):
    input_dim = len(hard_label)

    one_hot = torch.zeros((input_dim, 10)).cuda()

    one_hot.scatter_(1, hard_label.unsqueeze(1), 1)

    return one_hot


def cosine_similarity(matrix):
    dot_product = torch.matmul(matrix, matrix.t())

    norm = torch.norm(matrix, dim=1, keepdim=True)

    similarity = dot_product / torch.matmul(norm, norm.t())

    return similarity


def sum_intra_similarity(matrix):
    similarity_matrix = cosine_similarity(matrix)
    mask = torch.triu(torch.ones(similarity_matrix.shape), diagonal=1).cuda()

    intra_similarity_sum = torch.sum(similarity_matrix * mask)

    return intra_similarity_sum


def diversity_loss(inputs, targets):
    similarity = 0.0
    inputs = inputs.reshape(inputs.shape[0], -1)
    for i in range(torch.max(targets) + 1):
        temp_inputs = inputs[targets == i]
        similarity += sum_intra_similarity(temp_inputs)
    return similarity / inputs.shape[0]

def over_confidence_loss(s_prob):
    batch_size = s_prob.shape[0]
    return torch.std(s_prob, dim=1).sum() / batch_size

def data_aug(dataset):
    if dataset == "fmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation([90,180]),
            transforms.RandomCrop(size=(28, 28), padding=4),
        ])
        return transform,transform
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.RandomCrop(size=(28, 28), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation([90,180]),
        ])
        return transform,transform
    if dataset == 'svhn':
        transform = transforms.Compose(
            [augmentation.ColorJitter(0.2, 0.2, 0.2), augmentation.RandomChannelShuffle(p=0.5),
             augmentation.RandomGaussianNoise(p=0.2),
             augmentation.RandomSolarize(p=0.3),
             augmentation.RandomErasing(p=0.2),
             augmentation.RandomAffine(padding_mode='border', degrees=45),
             augmentation.RandomHorizontalFlip(),
             augmentation.RandomVerticalFlip(),
             augmentation.RandomPerspective(),
             augmentation.RandomInvert(p=0.2),
             # augmentation.RandomCrop(size=(32, 32), padding=4),
             ])
        aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(hue=0.3),
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(90, expand=True) if torch.rand(1) < 0.3 else transforms.RandomRotation(0)
            ])
        return aug, transform
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [augmentation.ColorJitter(0.2, 0.2, 0.2), augmentation.RandomChannelShuffle(p=0.5),
             augmentation.RandomGaussianNoise(p=0.2),
             augmentation.RandomSolarize(p=0.3),
             augmentation.RandomErasing(p=0.2),
             augmentation.RandomAffine(padding_mode='border', degrees=45),
             augmentation.RandomHorizontalFlip(),
             augmentation.RandomVerticalFlip(),
             augmentation.RandomPerspective(),
             augmentation.RandomInvert(p=0.2),
             augmentation.RandomCrop(size=(32, 32), padding=4),
             ])
        aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(hue=0.3),
                transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(90, expand=True) if torch.rand(1) < 0.3 else transforms.RandomRotation(0),
            ])
        return aug, transform



