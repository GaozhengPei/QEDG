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
from utils import ImagePool, reset_model, get_dataset, setup_seed, get_target_model, get_substitute_model, test_acc, write, test_robust, diversity_loss, data_aug,test_cohen_kappa,over_confidence_loss,free_aug,Imbalance_CrossEntropyLoss


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
        self.aug,self.transform = data_aug(args.dataset)

    def get_data(self):
        ### Retrieve all data
        datasets = self.data_pool.get_dataset()  
        global query
        query = len(datasets)
        print('The number of query is {}'.format(len(datasets)))
        with open('./{}.txt'.format(args.dataset), 'a') as f:
            f.write(str(len(datasets)) + '\n')
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.batch_size, shuffle=False,
            num_workers=4, drop_last=True)
        return self.data_loader

    def gen_data(self,sub_net):
        sub_net.eval()
        min_loss = 1e6
        best_inputs = None
        z = torch.randn(size=(self.batch_size, self.nz)).cuda()  #
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.batch_size,)).cuda()
        # print(targets)
        reset_model(self.generator)
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), 0.001)
        for it in range(args.g_steps):
            gen_optimizer.zero_grad()
            inputs = self.generator(z)
            inputs = self.aug(inputs)
            s_out = sub_net(inputs)
            s_prob = torch.softmax(s_out, dim=1)
            # print('Generate Acc ',torch.sum(torch.argmax(s_prob,dim=1)==targets)/self.batch_size)
            # print("top-1 prob",torch.max(s_prob,dim=1)[0])
            ###  over_confidence_loss
            oc_loss = over_confidence_loss(s_prob)
            ### classification loss
            class_loss = torch.nn.CrossEntropyLoss()(s_out, targets)  
            ### diversity loss
            div_loss = diversity_loss(inputs,targets)
            loss = class_loss + args.over_confidence_scale * oc_loss + args.div_scale * div_loss
            # print(class_loss,oc_loss,div_loss)
            # print(s_prob)
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
    parser.add_argument('--dataset', type=str, default='fmnist', help="name \
                        of dataset")
    parser.add_argument('--black_net', default='resnet34', type=str)
    parser.add_argument('--sub_net', default='resnet18', type=str)
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--g_steps', default=100, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=1024, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--query', default=30000, type=int,
                        help='The number of query')
    parser.add_argument('--free_aug', default=False, type=bool,
                        help='Free augment')        
    parser.add_argument('--balance_scale', default=2, type=float,
                        help='Free augment')             
    parser.add_argument('--over_confidence_scale', default=10, type=float,
                        help='The coefficient of over_confidence_loss')    
    parser.add_argument('--div_scale', default=0.1, type=float,
                        help='The coefficient of div_loss')   
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
            if args.free_aug:
                images = free_aug(images,substitute_outputs)
                substitute_outputs = sub_net(images)
            loss_ce = Imbalance_CrossEntropyLoss(substitute_outputs, labels,args.balance_scale)
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
    best_con = 0.0
    query = 0
    Query = args.query
    while query < Query:
        synthesizer.gen_data(sub_net)  # g_steps
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
