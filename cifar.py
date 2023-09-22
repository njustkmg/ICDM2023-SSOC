import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import numpy as np
from cifar_trainer import Trainer
import wandb
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):
    torch.set_num_threads(6)
    np.set_printoptions(threshold=np.inf)
    torch.autograd.set_detect_anomaly(True)
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args.seed = torch.randint(low=0, high=10000000, size=(1,))
    args.seed = 3407
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.save_path = os.path.join(args.save, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if args.dataset == "cifar10":
        args.image_size = 224
        args.feature_dim = 128
        args.init_num = 10
    else:
        args.image_size = 32
        args.feature_dim = 512
        args.init_num = 100

    init_classes = range(args.init_num)
    args.novel_num = int(args.init_num * args.novel_ratio)
    args.arg_novel_num = args.novel_num
    args.seen_num = args.init_num - args.novel_num
    args.seen_classes = range(args.seen_num)
    args.novel_classes = []
    for i in init_classes:
        if i not in args.seen_classes:
            args.novel_classes.append(i)

    project_name = args.dataset + "_" + str(args.label_ratio) + "_" + str(args.novel_ratio)
    wandb.init(
        project=project_name,
        config={
            "dataset": args.dataset,
            "lable_ratio": args.label_ratio,
            "novel_ratio": args.novel_ratio,
            "lr1": args.lr1,
            "lr2": args.lr2,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "layers": args.n_layers,
            "heads": args.n_heads,
            "img": args.image_size,
            "seed": args.seed,
            "a": args.a,
            "b": args.b,
            "c": args.c,
            "d": args.d,
            "th1": args.th1,
            "th2": args.th2
        }
    )

    trainer = Trainer(args, device)
    # trainer.load()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSOC')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--path', default='../data', help='dataset dir')
    parser.add_argument('--save', type=str, default='./results/')
    
    parser.add_argument('--novel_ratio', default=0.5, type=float)  # 0.1, 0.3, 0.5, 0.7, 0.9
    parser.add_argument('--label_ratio', default=0.5, type=float)  # 0.1, 0.3, 0.5
    parser.add_argument('--init_num', default=100, type=int)
    parser.add_argument('--seen_num', default=50, type=int)
    parser.add_argument('--seen_classes', default=[], type=list)
    parser.add_argument('--novel_num', default=50, type=int)
    parser.add_argument('--novel_classes', default=[], type=list)
    parser.add_argument('--arg_novel_num', default=50, type=int)
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--image_size', default=32, type=int)

    parser.add_argument('--epochs', type=int, default=500)    # cifar10: 200, cifar100: 500
    parser.add_argument('--lr1', type=float, default=0.0001)
    parser.add_argument('--lr2', type=float, default=0.0001)  # cifar10: 0.005, cifar100: 0.0001

    parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')  # cifar10: 128, cifar100: 512
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    
    parser.add_argument('--cluster', default=False, type=bool)

    # cifar10:  a=0.3, b=0.4, c=0.6, d=1
    # cifar100: a=10,  b=1,   c=1,   d=1
    parser.add_argument('--a', default=10, type=float)     # 0.3
    parser.add_argument('--b', default=1, type=float)     # 0.4
    parser.add_argument('--c', default=1, type=float)     # 0.6
    parser.add_argument('--d', default=1, type=float)       # 1

    parser.add_argument('--th1', default=0.7, type=float)
    parser.add_argument('--th2', default=0.8, type=float)

    args = parser.parse_args()
    main(args)
