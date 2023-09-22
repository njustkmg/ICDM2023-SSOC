import os
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import unsqueeze
from models import TransformerEncoder, resnet50
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
        num_class,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.num_class = num_class
        self.net = resnet50(num_classes=num_class).to(self.device)
        net_fp = torch.load('pretrain/simclr_imagenet_100.pth.tar')
        self.net.load_state_dict(net_fp, strict=False)

        self.encoder = TransformerEncoder(d_model=args.feature_dim, n_layers=args.n_layers, n_heads=args.n_heads).to(self.device)   # transformer
        self.encoder.apply(self.encoder.init_weights)
        self.tem = torch.full((args.batch_size, ), args.tem)
        self.tem = torch.tensor(self.tem, requires_grad=True)
        self.label_len = args.labeled_batch_size


    def forward(self, x, center, step):
        feature = self.net(x)
        center_new = self.encoder(feature, center)
        center_T = center_new.transpose(0, 1)
        cos = torch.mm(feature, center_T)
        cos.div_(self.tem[step])
        prob = F.softmax(cos, dim=-1)
        return cos, prob, feature, center_new.detach()


    def forward_test(self, x, center):
        feature = self.net(x)
        center_T = center.transpose(0, 1)
        cos = torch.mm(feature, center_T)
        prob = F.softmax(cos, dim=-1)
        return prob