import argparse
import os
import shutil
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from kornia import augmentation
from nets import Generator_2, GeneratorB
from utils import ImagePool, reset_model, get_dataset, setup_seed, get_target_model, get_substitute_model, test_acc, write, test_robust, diversity_loss, data_aug,new_loss,test_cohen_kappa


class Synthesizer():
    def __init__(self, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 batch_size, dataset):
        super(Synthesizer, self).__init__()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_pool = ImagePool(images_dir, dataset)
        self.generator = generator.cuda().train()
        # self.transform = transforms.Compose([# augmentation.ColorJitter(0.2,0.2,0.2),augmentation.RandomChannelShuffle(p=0.5), 
        #                 augmentation.RandomGaussianNoise(),
        #                 # augmentation.RandomSolarize(p=0.3),
        #                 augmentation.RandomErasing(),
        #                 augmentation.RandomAffine(padding_mode='border',degrees=45),
        #                 augmentation.RandomHorizontalFlip(),
        #                 augmentation.RandomVerticalFlip(),
        #                 augmentation.RandomPerspective(),
        #                 # augmentation.RandomInvert(p=0.2),
        #                 augmentation.RandomCrop(size=(self.img_size[-2],self.img_size[-1]),padding=4)])
        # self.aug =transforms.Compose(
        #     [
                # transforms.ColorJitter(brightness=0.5),
                # transforms.ColorJitter(hue=0.3),
                # transforms.RandomCrop(size=(32,32),padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.RandomRotation(90,expand=True) # if torch.rand(1) < 0.3 else transforms.RandomRotation(0)
        #     ])
        self.aug,self.transform = data_aug(args.dataset)

    def get_data(self):
        datasets = self.data_pool.get_dataset()  # 获取程序运行到现在所有的图片
        global query
        query = len(datasets)
        print('The number of query is {}'.format(len(datasets)))
        with open('./{}.txt'.format(args.dataset), 'a') as f:
            f.write(str(len(datasets)) + '\n')
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=True)
        return self.data_loader

    def gen_data(self):
        sub_net.eval()
        min_loss = 1e6
        best_inputs = None
        z = torch.randn(size=(self.batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.batch_size,)).cuda()
        reset_model(self.generator)
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), 0.001, betas=(0.5, 0.999))
        for it in range(args.g_steps):
            gen_optimizer.zero_grad()
            inputs = self.generator(z)
            inputs = self.aug(inputs)
            s_out = sub_net(inputs)
            s_prob = torch.softmax(s_out, dim=1)
            over_confidence_loss = torch.std(s_prob, dim=1).sum() / self.batch_size
            class_loss = torch.nn.CrossEntropyLoss()(s_out, targets)  # ce_loss
            div_loss = diversity_loss(inputs,targets)
            loss = class_loss + over_confidence_loss + 0.01 * div_loss
            # print(class_loss, over_confidence_loss)
            # print((torch.argmax(s_out, dim=1) == targets).sum())
            if min_loss > loss.item():
                min_loss = loss.item()
                best_inputs = inputs.data
            loss.backward()
            gen_optimizer.step()
        self.data_pool.add(best_inputs, self.transform, blackBox_net, sub_net, args.select_threshold)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50,
                        help="The number of rounds of training after the number of query is exhausted")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The learning rate of the substitute model')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--black_net', default='resnet34', type=str)
    parser.add_argument('--sub_net', default='resnet18', type=str)
    parser.add_argument('--lr_g', default=1e-2, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--g_steps', default=100, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=512, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--query', default=5000, type=int,
                        help='The number of query')
    parser.add_argument('--sub_model_weight_path', default='./sub_model_weight', type=str,
                        help='The path of the substitute model weight')
    parser.add_argument('--select_threshold', default=0.7, type=float,
                        help='select_threshold')
    parser.add_argument('--save_images_dir', default='./images_generated', type=str, \
                        help='the path you want to save the pic generated by generator')
    args = parser.parse_args()
    return args


def sub_train(synthesizer, optimizer):
    sub_net.train()
    data = synthesizer.get_data()
    for _ in range(20):
        for images, labels in data:
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            substitute_outputs = sub_net(images)
            # loss_ce = new_loss(substitute_outputs, labels)
            loss_ce = F.cross_entropy(substitute_outputs, labels)
            # loss_l1 = torch.nn.L1Loss()(images,labels)
            if  loss_ce > 0:
                loss = loss_ce
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    args = args_parser()
    with open('./{}.txt'.format(args.dataset), 'a') as f:
        f.write("====================================\n")
        f.write("The time now is:{}".format(time.localtime())+'\n')
        f.write('target model:{} substitute_model:{}'.format(args.black_net,args.sub_net)+'\n')
    images_dir = args.save_images_dir
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    else:
        os.mkdir(images_dir)
    if not os.path.exists(args.sub_model_weight_path):
        os.mkdir(args.sub_model_weight_path)
    setup_seed(args.seed)
    _, test_loader = get_dataset(args.dataset)

    sub_net = get_substitute_model(args.dataset, args.sub_net)
    blackBox_net = get_target_model(args.dataset, args.black_net)

    acc, _ = test_acc(blackBox_net, test_loader)
    print('target model:',args.black_net)
    print('substitute model',args.sub_net)
    with open('./{}.txt'.format(args.dataset), 'a') as f:
        f.write("Accuracy of the black-box model:{:.3} % \n".format(acc))
    print("Accuracy of the black-box model:{:.3} %".format(acc))
    acc, _ = test_acc(sub_net, test_loader)
    con = test_cohen_kappa(sub_net, blackBox_net, test_loader)
    asr = test_robust(sub_net, blackBox_net, test_loader)
    write(acc, con, con, asr, args.dataset)
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        image_size = [1, 28, 28]
        num_class = 10
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        image_size = [3, 32, 32]
        num_class = 10
    elif args.dataset == 'cifar100':
        image_size = [3, 32, 32]
        num_class = 100
    elif args.dataset == 'tiny-imagenet':
        image_size = [3, 64, 64]
        num_class = 200
    generator = Generator_2(nz=args.nz, ngf=64, img_size=image_size[-1], nc=image_size[0]).cuda()

    synthesizer = Synthesizer(generator,
                              nz=args.nz,
                              num_classes=num_class,
                              img_size=image_size,
                              iterations=args.g_steps,
                              lr_g=args.lr_g,
                              batch_size=args.batch_size,
                              dataset=args.dataset)
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    optimizer = optim.SGD(sub_net.parameters(), args.lr, args.momentum)
    sub_net.train()
    best_con = 0
    query = 0
    Query = args.query
    while query < Query:
        synthesizer.gen_data()  # g_steps
        sub_train(synthesizer, optimizer)
        acc, _ = test_acc(sub_net, test_loader)
        con = test_cohen_kappa(sub_net, blackBox_net, test_loader)
        asr = test_robust(sub_net, blackBox_net, test_loader)

        if con > best_con:
            best_con = con
            torch.save(sub_net.state_dict(),
                       args.sub_model_weight_path + '/{}_{}_{}.pth'.format(args.sub_net,args.black_net, args.dataset))
        write(acc, con,best_con, asr, args.dataset)
    for epoch in range(args.epochs):
        with open('./{}.txt'.format(args.dataset), 'a') as f:
            f.write("epoch:{}/{} \n".format(epoch, args.epochs))
        print("epoch:{}/{} \n".format(epoch, args.epochs))
        sub_train(synthesizer, optimizer)
        acc, _ = test_acc(sub_net, test_loader)
        con = test_cohen_kappa(sub_net, blackBox_net, test_loader)
        asr = test_robust(sub_net, blackBox_net, test_loader)
        if con > best_con:
            best_con = con
            torch.save(sub_net.state_dict(),
                       args.sub_model_weight_path + '/{}_{}_{}.pth'.format(args.sub_net,args.black_net, args.dataset))
        write(acc, con,best_con, asr, args.dataset)
