import os
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import unsqueeze
from models import TransformerEncoder, get_resnet, Network, resnet18, resnet50
import matplotlib.pyplot as plot
from sklearn import manifold
import torch.nn.functional as F
import math
import numpy as np
import torchvision


class OUR(nn.Module):
    def __init__(
        self,
        args,
        device,
        num_class
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.num_class = num_class

        if args.dataset == "cifar10":
            self.res = get_resnet('resnet34').to(self.device)
            self.net = Network(self.res, args.feature_dim, num_class).to(self.device)
            net_fp = "pretrain/checkpoint_1000_224.tar"
            self.net.load_state_dict(torch.load(net_fp, map_location=device.type)['net'])
        else:
            self.net = resnet18(num_classes=num_class).to(self.device)
            net_fp = torch.load('pretrain/simclr_cifar_100.pth.tar')
            self.net.load_state_dict(net_fp, strict=False)

        self.encoder = TransformerEncoder(d_model=args.feature_dim, n_layers=args.n_layers, n_heads=args.n_heads).to(self.device)
        self.encoder.apply(self.encoder.init_weights)
        if args.dataset == "cifar10":
            self.tem = torch.pow((10.0 / 11.0), torch.arange(0, self.args.epochs / 2)) + 1
        else:
            self.tem = (2.0 - torch.pow((10.0 / 11.0), torch.arange(0, self.args.epochs / 2))) / 2.0
        self.tem = torch.tensor(self.tem, requires_grad=True)
        self.label_len = args.labeled_batch_size


    def forward(self, x, center, step):
        if self.args.image_size == 224:
            feature = self.net.forward_feature(x)
        else:
            feature = self.net(x)
        center_new = self.encoder(feature, center)
        center_T = center_new.transpose(0, 1)
        cos = torch.mm(feature, center_T)
        cos[:self.label_len].div_(self.tem[(step - 1) // 2])
        prob = F.softmax(cos, dim=-1)
        return cos, prob, feature, center_new.detach()


    def forward_test(self, x, center):
        if self.args.image_size == 224:
            feature = self.net.forward_feature(x)
        else:
            feature = self.net(x)
        center_T = center.transpose(0, 1)
        cos = torch.mm(feature, center_T)
        prob = F.softmax(cos, dim=-1)
        return prob