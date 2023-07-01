from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models.mobilefacenet import MobileFaceNet
import torchvision
from torchvision import transforms as T
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
from datetime import datetime


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def save_optimizer(optimizer, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(optimizer.state_dict(), save_name)
    return save_name

if __name__ == '__main__':
    runtime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    s = 64
    m = 0.2
    opt = Config()

    log_file1 = open(os.path.join('log', '_s=' + str(s) + '_m=' + str(m) + "batch_size=" + str(opt.train_batch_size) + "testing.txt"), "w", encoding="utf-8")
    log_file1.write("epoch\ttest_acc\n")
    log_file2 = open(os.path.join('log', '_s=' + str(s) + '_m=' + str(m) + "batch_size=" + str(opt.train_batch_size) + "training.txt"), "w", encoding="utf-8")

    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_transforms = T.Compose([
        T.RandomCrop(opt.input_shape[1:]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(opt.train_root, transform=train_transforms)
    # train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    # train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    # trainloader = data.DataLoader(train_dataset,
    #                               batch_size=opt.train_batch_size,
    #                               shuffle=True,
    #                               num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    elif opt.backbone == 'mobilefacenet':
        model = MobileFaceNet(512).to(torch.device("cuda:0") if torch.cuda.is_available() else "cpu")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.load_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(model)

    model.conv1.requires_grad = False
    model.conv2_dw.requires_grad_ = False
    model.conv_23.requires_grad = False
    model.conv_3.requires_grad = False
    model.conv_34.requires_grad = False
    model.conv_4.requires_grad = False
    model.conv_45.requires_grad = False
    model.conv_5.requires_grad = False
    model.conv_6_dw.requires_grad = True
    model.linear.requires_grad = True
    model.bn.requires_grad = True

    
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=s, m=m)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=s, m=m, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.load_optimizer:  
      optimizer_dict = optimizer.state_dict()
      pretrained_optimizer_dict = torch.load(opt.checkpoints_optimizer_path)
      pretrained_optimizer_dict = {k: v for k, v in pretrained_optimizer_dict.items() if k in optimizer_dict}
      optimizer_dict.update(pretrained_optimizer_dict)
      optimizer.load_state_dict(optimizer_dict)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch + 1):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(),
                                                                                   acc))
                log_file2.write('{} train epoch {} iter {} {} iters/s loss {} acc {}\n'.format(time_str, i, ii, speed, loss.item(),
                                                                                   acc))
                if opt. display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone + '_s=' + str(s) + '_m=' + str(m) + "batch_size=" + str(opt.train_batch_size) , i)
            save_optimizer(model, opt.checkpoints_optimizer_save_path, opt.optimizer + '_s=' + str(s) + '_m=' + str(m) + "batch_size=" + str(opt.train_batch_size) , i)

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        log_file1.write("%s\t%.3f\n" \
                       % (i, acc))
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')
