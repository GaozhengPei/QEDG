import argparse
import os

import torch
from torch import nn
from nets import CNN
from utils import get_dataset,test

from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack, GradientSignAttack


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='FGSM',
                        help="Attack method")
    parser.add_argument('--net', type=str, default='CNN',
                        help='The network of substitute')
    parser.add_argument('--weight_dir', type=str, default='./sub_model_weight',
                        help='state_dict_dir_saved')
    parser.add_argument('--dataset', type=str, default='fmnist', help="name of dataset")
    parser.add_argument('--attack_step_size', default=0.01, type=float)
    parser.add_argument('--attack_max', default=0.3, type=float)
    parser.add_argument('--target', default=False, type=bool)
    args = parser.parse_args()

    return args


def test_robust(attack, sub_net, dataset):
    sub_net.eval()
    _, test_loader = get_dataset(dataset)
    if attack == 'FGSM':
        adversary = GradientSignAttack(
            sub_net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.15,
            targeted=args.target)
    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(sub_net,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.15,nb_iter=200,eps_iter=0.01,clip_min=0.0, clip_max=1.0,targeted=args.target)
    if attack == 'PGD':
        adversary = LinfPGDAttack(sub_net,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.15,nb_iter=200,eps_iter=0.01,clip_min=0.0, clip_max=1.0,targeted=args.target)
    correct = 0
    total = 0
    for each in test_loader:
        images = each[0].cuda()
        labels = each[1].cuda()
        idx = torch.argmax(sub_net(images),dim=1)==labels
        total += idx.sum()
        adv_images = adversary.perturb(images[idx], labels[idx])
        predict = torch.argmax(sub_net(adv_images),dim=1)
        correct += (predict != labels[idx]).sum()
    return correct / total

if __name__ == '__main__':
    args = args_parser()
    net = CNN().cuda()
    weight_path = os.path.join(args.weight_dir, args.net + '_' + args.dataset + '.pth')
    state_dict = torch.load(weight_path)
    net.load_state_dict(state_dict)
    asr = test_robust('PGD', net, args.dataset)
    _, test_loader = get_dataset(args.dataset)
    acc,_ = test(net,test_loader)
    print(acc)
    print(asr)
