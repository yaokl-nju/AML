import torch
import torch.nn.functional as F
from nn.GNNs import *
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
        self.gnn_net = GNN(args).to(self.args.device)
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
        graph, feat, labels, nlinks, ids_map, feat_v, feat_d = dataset.sample('train')
        start_time = time.time()
        self.train()
        self.optimizer.zero_grad()
        self.zero_grad()
        logits = self.gnn_net(feat, graph, nlinks, ids_map, feat_v, feat_d)
        if self.args.lossfunc == 'pairwise':
            mask = labels > 0
            loss = hinge_auc_loss(logits[mask], logits[torch.logical_not(mask)], self.args.num_neg)
        else:
            loss = BCEWithLogitsLoss()(logits, labels.to(torch.float32))
        loss.backward()
        self.optimizer.step()
        self.T2 += (time.time() - start_time)
        return {'loss': loss.item()}

    @torch.no_grad()
    def evaluation_logits(self, dataset):
        self.eval()
        ### step 1: cal node representation
        def get_emb_gnn(net, dataset, axis, phase):
            xconv = []
            for i in range(dataset.batch['node']):
                graph, feat, ids_map = dataset.sample_node(phase)
                xconv.append(net.forward_gnn(feat, graph, ids_map, axis))
            xconv = torch.cat(xconv)
            torch.cuda.empty_cache()
            return xconv

        def get_emb_mlp(net, dataset, phase):
            xmlp = []
            for i in range(dataset.batch['node']):
                feat_v, feat_d = dataset.sample_node(phase, False)
                xmlp.append(net.forward_mlp(None, feat_v, feat_d))
            xmlp = torch.cat(xmlp)
            torch.cuda.empty_cache()
            return xmlp

        def get_predict(net, dataset, x_l, x_r, phase, reverse):
            logits, labels = [], []
            for i in range(dataset.batch[phase]):
                links_i, labels_i = dataset.sample_eval(phase)
                axis = 1 if reverse else 0
                src, obj = links_i[axis], links_i[1 - axis]
                logits_i = net.forward_predict(x_l[src], x_r[obj]).data
                logits_i = (logits_i + net.forward_predict(x_r[src], x_l[obj]).data) / 2 if not self.args.directed else logits_i
                logits.append(logits_i)
                # logits.append(net.forward_predict(x_r[src], x_r[obj]).data)
                labels.append(labels_i)
                torch.cuda.empty_cache()
            logits = torch.cat(logits).view(-1).cpu()
            labels = torch.cat(labels).view(-1).cpu()
            return logits, labels

        x_l_v = get_emb_gnn(self.gnn_net, dataset, 'l', 'valid')
        x_l_t = get_emb_gnn(self.gnn_net, dataset, 'l', 'test') if self.args.use_valedges_as_input else x_l_v

        if self.args.strategy == 'symmetric':
            # if self.args.directed:
            #     x_r_v = get_emb_gnn(dataset, 'r', 'valid')
            #     x_r_t = get_emb_gnn(dataset, 'r', 'test')
            # else:
            x_r_v, x_r_t = x_l_v, x_l_t
        else:
            x_r_v = get_emb_mlp(self.gnn_net, dataset, 'valid')
            x_r_t = get_emb_mlp(self.gnn_net, dataset, 'test') if self.args.use_valedges_as_input else x_r_v

            ### model homophily
            x_l_v = x_l_v + x_r_v
            x_l_t = x_l_t + x_r_t

        logits_v, labels_v = get_predict(self.gnn_net, dataset, x_l_v, x_r_v, 'valid', self.args.reverse)
        logits_t, labels_t = get_predict(self.gnn_net, dataset, x_l_t, x_r_t, 'test', self.args.reverse)

        pos_pred_v = logits_v[labels_v == 1]
        neg_pred_v = logits_v[labels_v == 0]
        pos_pred_t = logits_t[labels_t == 1]
        neg_pred_t = logits_t[labels_t == 0]
        return pos_pred_v, neg_pred_v, logits_v, labels_v, pos_pred_t, neg_pred_t, logits_t, labels_t

    @torch.no_grad()
    def evaluation(self, dataset):
        v_pos_pred, v_neg_pred, v_logits, v_labels, t_pos_pred, t_neg_pred, t_logits, t_labels = self.evaluation_logits(dataset)

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
        node_emb = self.gnn_net.get_node_embedding()
        return node_emb.cpu()

