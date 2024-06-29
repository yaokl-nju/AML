import torch
import torch.nn.functional as F
# from nn.GNNs import *
from nn.MLPs import *
import gc
import time
from sklearn.metrics import f1_score
from ogb.linkproppred import Evaluator
from utils.init_func import dropout, row_norm
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score
from Trainer.loss_func import *


def get_optimizer(name, params, lr, lamda=0):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=lamda, momentum=0.9)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=lamda)
    elif name == 'adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=lamda)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=lamda)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=lamda)
    elif name == 'adamax':
        return torch.optim.Adamax(params, lr=lr, weight_decay=lamda)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def get_scheduler(optimizer, name, epochs):
    if name == 'Mstep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3*2, epochs//4*3], gamma = 0.1)
    elif name == 'Expo':
        gamma = (1e-6)**(1.0/epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)
    elif name == 'Cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-16, last_epoch=-1)
    else:
        raise ValueError('Wrong LR schedule!!')

class Trainer_ogb(torch.nn.Module):
    def __init__(self, args):
        super(Trainer_ogb, self).__init__()
        self.args = args
        self.gnn_net = MLPs(args).to(self.args.device)
        self.optimizer = get_optimizer(
            args.optimizer,
            self.gnn_net.parameters(),
            args.lr,
            args.lamda
        )
        self.scheduler = get_scheduler(self.optimizer, args.lrschedular, args.epochs)

        self.epsilon = 1e-7
        self.T2 = 0.0
        self.evaluator = Evaluator(args.dataset)

    def update(self, dataset):
        labels, nlinks, ids_map, feat = dataset.sample('train')
        start_time = time.time()
        self.train()
        self.optimizer.zero_grad()
        self.zero_grad()
        logits = self.gnn_net(feat, nlinks, ids_map)
        if self.args.lossfunc == 'pairwise':
            shift = logits.size(0) // 2
            loss = hinge_auc_loss(logits[:shift].view(-1), logits[shift:].view(-1), 1)
        else:
            loss = BCEWithLogitsLoss()(logits.view(-1), labels.view(-1).to(torch.float32))
        loss.backward()
        self.optimizer.step()
        self.T2 += (time.time() - start_time)
        return {'loss': loss.item(), 'acc': 0.0}

    @torch.no_grad()
    def evaluation_logits(self, dataset, phase):
        logits, labels = [], []
        for i in range(dataset.batch[phase]):
            self.eval()
            labels_i, nlinks, ids_map, feat = dataset.sample(phase)
            logits_i = self.gnn_net(feat, nlinks, ids_map)
            logits.append(logits_i.data)
            labels.append(labels_i)
            torch.cuda.empty_cache()
        logits = torch.cat(logits).view(-1).cpu()
        labels = torch.cat(labels).view(-1).cpu()
        torch.cuda.empty_cache()

        pos_pred = logits[labels == 1]
        neg_pred = logits[labels == 0]
        return pos_pred, neg_pred, logits, labels

    @torch.no_grad()
    def evaluation(self, dataset):
        v_pos_pred, v_neg_pred, v_logits, v_labels = self.evaluation_logits(dataset, 'valid')
        t_pos_pred, t_neg_pred, t_logits, t_labels = self.evaluation_logits(dataset, 'test')

        v_loss = BCEWithLogitsLoss()(v_logits.to(torch.float32), v_labels.to(torch.float32))
        t_loss = BCEWithLogitsLoss()(t_logits.to(torch.float32), t_labels.to(torch.float32))
        if self.args.eval_metric == 'hits':
            results = Trainer_ogb.evaluate_hits(self.evaluator, v_pos_pred, v_neg_pred, t_pos_pred, t_neg_pred)
        elif self.args.eval_metric == 'mrr':
            results = Trainer_ogb.evaluate_mrr(self.evaluator, v_pos_pred, v_neg_pred, t_pos_pred, t_neg_pred)
        elif self.args.eval_metric == 'auc':
            results = Trainer_ogb.evaluate_auc(v_logits, v_labels, t_logits, t_labels)
        else:
            assert False
        return results, {'v_loss': v_loss.item(), 't_loss': t_loss.item()}

    @staticmethod
    def evaluate_hits(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_val_pred,
                'y_pred_neg': neg_val_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']
            results[f'Hits@{K}'] = (valid_hits, test_hits)
        return results

    @staticmethod
    def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
        neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
        neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
        results = {}
        valid_mrr = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })['mrr_list'].mean().item()

        test_mrr = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })['mrr_list'].mean().item()
        results['MRR'] = (valid_mrr, test_mrr)
        return results

    @staticmethod
    def evaluate_auc(val_pred, val_true, test_pred, test_true):
        valid_auc = roc_auc_score(val_true, val_pred)
        test_auc = roc_auc_score(test_true, test_pred)
        results = {}
        results['AUC'] = (valid_auc, test_auc)
        return results

    @torch.no_grad()
    def get_node_embedding(self):
        x_in = self.gnn_net.get_node_embedding(None)
        return x_in.cpu()
