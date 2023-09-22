import os
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm
from models.ssoc_cifar import OUR
import torch.optim as optim
from itertools import cycle
import torch.nn.functional as F
from matplotlib import test
import matplotlib.pyplot as plot
import matplotlib.colors as mcolors
from genericpath import exists
from pickletools import optimize
from sklearn import metrics
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFdr
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from utils import cluster_acc, accuracy, get_auc_roc, op_toexcel, bce_loss, entropy
from scipy.optimize import linear_sum_assignment as linear_assignment
import hypertools as hyp
from data.open_world_cifar import OPENWORLDCIFAR10, OPENWORLDCIFAR100, TransformTwice
from data.transform import get_transform
import time

class Trainer(nn.Module):
    def __init__(
        self,
        args,
        device,
    ):
        super().__init__()
        self.args = args
        self.device = device

        dict_transform = get_transform(args.dataset, args.image_size)
        datamap = {'cifar10': OPENWORLDCIFAR10, 'cifar100': OPENWORLDCIFAR100}
        if args.cluster is False:
            trans = TransformTwice(dict_transform['train'])
        else:
            trans = dict_transform['train']

        train_label_set = datamap[args.dataset](root=args.path, labeled=True, labeled_num=args.seen_num, 
                                                labeled_ratio=args.label_ratio, download=True, transform=dict_transform['train'])
        train_unlabel_set = datamap[args.dataset](root=args.path, labeled=False, labeled_num=args.seen_num, 
                                                labeled_ratio=args.label_ratio, transform=trans, unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datamap[args.dataset](root=args.path, labeled=False, labeled_num=args.seen_num, 
                                                labeled_ratio=args.label_ratio, transform=dict_transform['test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.labeled_len = len(train_label_set)
        args.unlabeled_len = len(train_unlabel_set)
        args.update_ratio = 1.0 * args.batch_size / (args.labeled_len + args.unlabeled_len)
 
        self.labeled_batch_size = int(args.batch_size * args.labeled_len / (args.labeled_len + args.unlabeled_len))
        args.labeled_batch_size = self.labeled_batch_size
        self.train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=self.labeled_batch_size, shuffle=True,
                                                        num_workers=0, pin_memory=True, drop_last=True)
        self.train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - self.labeled_batch_size, 
                                                        shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=0)


        self.num_class = args.seen_num + args.arg_novel_num
        self.model = OUR(args, device, num_class=self.num_class).cuda()

        self.optim1 = optim.Adam(self.model.net.parameters(), lr=args.lr1, betas=(0.9, 0.99))
        self.optim2 = optim.Adam(self.model.encoder.parameters(), lr=args.lr2, betas=(0.9, 0.99))
        self.sche1 = optim.lr_scheduler.CosineAnnealingLR(self.optim1, T_max=args.epochs)  # 15
        self.sche2 = optim.lr_scheduler.CosineAnnealingLR(self.optim2, T_max=args.epochs)  # 50  args.epochs
        
        if args.cluster:
            center = self.cluster()
            exit()
        else:
            center = torch.load(os.path.join("{}/{}_{}.pt".format(args.save_path, args.image_size, args.feature_dim)))

        self.args.save_path = os.path.join(args.save_path, str(int(args.label_ratio * 10)) + "_" + str(int(args.novel_ratio * 10)))
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        self.center = torch.tensor(center).cuda().float()
        self.init_center = F.normalize(self.center)
        self.epoch = 0
        self.best_acc = 0

    def train(self):
        torch.set_printoptions(profile="default")
        while self.epoch < self.args.epochs:
            self.epoch += 1
            self.feature = []
            self.lbs = np.array([])
            tq = tqdm(self.train_label_loader, ncols=85)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)
            loss_all = 0
            loss_s = 0
            loss_p = 0
            loss_bce = 0

            for idx, (x, y) in enumerate(tq):
                self.model.train()
                (ux1, ux2), uy = next(unlabel_loader_iter)
                x, y, ux1, ux2, uy = x.cuda(), y.cuda(), ux1.cuda(), ux2.cuda(), uy.cuda()
                label_len = x.shape[0]
                unlabel_len = ux1.shape[0]

                cos, prob, feature, center = self.model(torch.cat([x, ux1, ux2], 0), self.center, self.epoch)
                max_prob, pseudo_lb = torch.max(prob, dim=-1)

                # [label data, unlabel data1]
                cos = cos[:(label_len + unlabel_len)]
                prob = prob[:(label_len + unlabel_len)]
                feature = feature[:(label_len + unlabel_len)]
                max_prob = max_prob[:(label_len + unlabel_len)]
                pseudo_lb = torch.cat([pseudo_lb[:label_len], pseudo_lb[-unlabel_len:]], 0)

                self.center = torch.add(self.center, center)
                self.now_center = F.normalize(self.center)

                # self.feature = self.feature + F.normalize(feature.detach()).cpu().numpy().tolist()
                # self.lbs = np.append(self.lbs, torch.cat([y, uy], 0).cpu().numpy())

                # ce loss
                ls = F.cross_entropy(prob[:label_len], y)
                loss_s += ls
                mask = max_prob[label_len:].ge(self.args.th1).float()
                lp = torch.sum(F.cross_entropy(prob[label_len:], pseudo_lb[label_len:], reduction='none') * mask) / torch.sum(mask)
                loss_p += lp

                # pair loss
                sim = torch.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=-1)
                gt = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).int()
                sim[:label_len, :label_len] = gt
                prob_sim = torch.cosine_similarity(cos.unsqueeze(1), cos.unsqueeze(0), dim=-1)
                mk1 = max_prob.ge(self.args.th2).float()
                tmp = mk1.expand(self.args.batch_size, -1)
                mk2 = tmp + tmp.T
                one = torch.ones_like(mk2)
                zero = torch.zeros_like(mk2)
                mask1 = torch.where(mk2 >= 2.0, one, zero)
                mask1 = mask1.int()
                mask2 = mask1 | (sim.int())
                lbce = torch.sum(bce_loss(prob_sim, sim) * mask2) / torch.sum(mask2)
                loss_bce += lbce

                # entropy regularization
                entropy_loss = entropy(torch.mean(prob, 0))
                loss = self.args.a * ls + self.args.b * lbce + self.args.c * lp - self.args.d * entropy_loss
                loss_all += loss

                self.optim1.zero_grad()
                self.optim2.zero_grad()
                loss.backward()
                self.optim1.step()
                self.optim2.step()

                tq.set_description("Epoch: {} | Loss: {}".format(self.epoch, loss.item()))
            wandb.log({"loss_epoch": loss_all, "loss_bce": loss_bce, "loss_s": loss_s, "loss_p": loss_p})

            tq.close()
            self.sche1.step()
            self.sche2.step()
            if self.epoch % self.args.eval_step == 0:
                result = self.test()
                op_toexcel(result, os.path.join(self.args.save_path, "log.xlsx"))
                # if self.epoch % 20 == 0:
                #     self.draw()
                if result[11] > self.best_acc:
                    self.best_acc = result[11]
                    self.save()
        wandb.finish()


    def test(self):
        self.model.eval()
        targets = np.array([])
        preds = np.array([])
        probs = []

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.test_loader):
                x, y = x.cuda(), y.cuda()
                prob = self.model.forward_test(x, self.center)
                max_prob, pseudo_lb = torch.max(prob, dim=-1)

                targets = np.append(targets, y.cpu().numpy())
                preds = np.append(preds, pseudo_lb.cpu().numpy())
                probs += prob.cpu().tolist()

            targets = targets.astype(int)
            preds = preds.astype(int)
            probs = np.array(probs)

            seen_mask = targets < self.args.seen_num
            unseen_mask = ~seen_mask
            all_acc = cluster_acc(preds, targets)
            all_error = 1.0 - all_acc
            all_auc = get_auc_roc(probs, targets)

            seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
            unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
            unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
            print('------Test------')
            print('  seen_acc {:.4f}'.format(seen_acc))
            print('unseen_acc {:.4f}'.format(unseen_acc))
            print('       acc {:.4f}'.format(all_acc))
            print('     error {:.4f}'.format(all_error))
            print('     auroc {:.4f}'.format(all_auc))
            
            result = [self.args.dataset, self.args.batch_size, self.args.lr1, self.args.lr2, self.epoch, 
                    self.args.label_ratio, self.args.novel_ratio, str(self.args.novel_classes),
                    seen_acc * 100, unseen_acc * 100, unseen_nmi * 100, all_acc * 100, all_error * 100, all_auc * 100]

            wandb.log({"seen_acc": seen_acc, "unseen_acc": unseen_acc, "acc": all_acc, "auroc": all_auc})
            return result


    def save(self):
        out = os.path.join(self.args.save_path, "best.tar")
        data = {
                'net': self.model.state_dict(),
                'optim1': self.optim1.state_dict(),
                'optim2': self.optim2.state_dict(),
                'sche1': self.sche1.state_dict(),
                'sche2': self.sche2.state_dict(),
                'epoch': self.epoch,
                'center': self.center,
                'best_acc': self.best_acc,
            }
        torch.save(data, out)
        

    def load(self):
        out = os.path.join(self.args.save_path, "best.tar")
        data = torch.load(out, map_location=self.device)
        self.model.load_state_dict(data['net'])
        self.optim1.load_state_dict(data['optim1'])
        self.optim2.load_state_dict(data['optim2'])
        self.sche1.load_state_dict(data['sche1'])
        self.sche2.load_state_dict(data['sche2'])
        self.epoch = data['epoch']
        self.center = data['center']
        self.best_acc = data['best_acc']


    def draw(self):
        lent = len(self.feature)
        X = np.array(self.feature + self.init_center.detach().cpu().numpy().tolist() + self.now_center.detach().cpu().numpy().tolist())
        y = list(map(int, np.array(self.lbs)))

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)  # n_components=2降维为2维并且可视化
        X_tsne = tsne.fit_transform(X)
        x_min, x_max = X_tsne.min(), X_tsne.max()
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plot.figure(figsize=(15, 15))

        colors = list(mcolors.CSS4_COLORS.keys())

        for i in range(X_norm.shape[0]):
            if i >= lent:
                if i - lent < self.num_class:
                    plot.text(X_norm[i, 0], X_norm[i, 1], str(i - lent), color='k', fontdict={'weight': 'bold', 'size': 20})
                else:
                    plot.text(X_norm[i, 0], X_norm[i, 1], str(i - lent - self.num_class), color='b', fontdict={'weight': 'bold', 'size': 20})
            else:
                plot.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})

        plot.savefig(os.path.join(self.args.save_path, str(self.epoch) + ".png"))


    def cluster(self):
        with torch.no_grad():
            feature = []
            lbs = np.array([])
            for idx, (x, y) in enumerate(tqdm(self.train_label_loader, ncols=85)):
                x = x.to(self.device)
                if self.args.image_size == 224:
                    output = self.model.net.forward_feature(x)
                else:
                    output = self.model.net(x)
                feature += output.cpu().tolist()
                lbs = np.append(lbs, y.numpy())
            
            for idx, (x, y) in enumerate(tqdm(self.train_unlabel_loader, ncols=85)):
                x = x.to(self.device)
                if self.args.image_size == 224:
                    output = self.model.net.forward_feature(x)
                else:
                    output = self.model.net(x)
                feature += output.cpu().tolist()
                lbs = np.append(lbs, y.numpy())

            lent = len(feature)
            # feature = torch.tensor(feature, dtype=torch.float32)
            # feature = F.normalize(feature, dim=1)
            # feature = feature.tolist()

            kmeans = KMeans(n_clusters=self.num_class, n_init=30, max_iter=1000, tol=0.000001).fit(feature)
            center = kmeans.cluster_centers_
            pred = kmeans.labels_
            pred = pred.astype(int)
            lbs = lbs.astype(int)
            mp = np.zeros((self.num_class, self.num_class), dtype=int)

            print(self.args.labeled_len)
            for i in range(self.args.labeled_len):
                mp[pred[i]][lbs[i]] += 1
            ind = linear_assignment(mp.max() - mp)
            ind = np.vstack(ind).T
            lb2pre = {j: i for i, j in ind}
            print(lb2pre)
            center_new = []
            for i in range(self.num_class):
                center_new.append(center[lb2pre[i]])

            X = np.array(feature + center_new)
            y = list(map(int, np.array(lbs)))
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)  # n_components=2降维为2维并且可视化
            X_tsne = tsne.fit_transform(X)
            x_min, x_max = X_tsne.min(), X_tsne.max()
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plot.figure(figsize=(15, 15))
            colors = list(mcolors.CSS4_COLORS.keys())
            for i in range(X_norm.shape[0]):
                if i >= lent:
                    plot.text(X_norm[i, 0], X_norm[i, 1], str(i - lent), color='k', fontdict={'weight': 'bold', 'size': 25})
                else:
                    plot.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})

            path = self.args.save_path + "/cluster.png"
            plot.savefig(path)

            torch.save(center_new, os.path.join("{}.pt".format(self.args.save_path + "/center")))
            return center_new

