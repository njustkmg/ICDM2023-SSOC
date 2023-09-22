from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import random


class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_transform(dataset, image_size):
    if dataset == "cifar10":
        dict_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8 * 0.5, 0.8 * 0.5, 0.8 * 0.5, 0.2 * 0.5)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(image_size, image_size)),
                transforms.ToTensor(),
            ]),
        }
    elif dataset == "cifar100":
        dict_transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        }
    else:
        dict_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    return dict_transform
