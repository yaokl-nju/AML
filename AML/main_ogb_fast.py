from Trainer.GNN_Trainer_ogb import *
from datasets.ogbl_dataset import ogbl_dataset
from datasets import _Sampler
from Trainer.early_stopping_ogb import *
import time
import gc
import itertools
import random
# from tensorboardX import SummaryWriter
from parse_conf import *
import joblib

dataset = ogbl_dataset(args, name=args.dataset, root=args.root)
args.num_nodes = dataset.num_nodes
args.num_features = dataset.num_feature
args.batch_tr = dataset.batch['train']
args.batch_val = dataset.batch['valid']
args.batch_te = dataset.batch['test']

def get_loggers():
    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.repeat, args),
            'Hits@50': Logger(args.repeat, args),
            'Hits@100': Logger(args.repeat, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.repeat, args),
        }
    elif args.eval_metric == 'auc':
        loggers = {
            'AUC': Logger(args.repeat, args),
        }
    else:
        assert False
    return loggers

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

lamdas = [args.lamda]
lrs = [args.lr]
drops = [args.drop]

results = []
for drop, lr, lamda in itertools.product(drops, lrs, lamdas):
    args.lamda = lamda
    args.lr = lr
    args.drop = drop
    acc_te_i = []

    loggers = get_loggers()
    for i in range(args.repeat):
        t_total = time.time()
        loss_val_j, acc_val_j = [], []

        model = Trainer_ogb(args)
        early_stopping = EarlyStopping(model, **stopping_args)
        dataset.reset_iter()

        sampler_tr = _Sampler(dataset, phase='train', buffer_size=50)
        sampler_node = _Sampler(dataset, phase='node', buffer_size=20 if args.method_te != 'FULL' else 1)
        print("lamda={}, lr={}, drop={}".format(
            str(args.lamda), str(args.lr), str(args.drop)))

        train_time = []
        for j in range(args.epochs):
            start_time = time.time()
            loss_k, loss_ratio, reg_ratio = 0., 0.0, 0.0
            for k in range(args.batch_tr):
                temp_lo = model.update(dataset)
                loss_k += temp_lo['loss']
            loss_k /= args.batch_tr

            if args.dataset != 'ogbl-ppa':
                temp_results, temp_loss = model.evaluation(dataset)
                for key, result in temp_results.items():
                    loggers[key].add_result(i, result)

            train_time.append(time.time() - start_time)
            log_step = args.eval_step
            if (j + 1) % log_step == 0:
                if args.dataset == 'ogbl-ppa':
                    temp_results, temp_loss = model.evaluation(dataset)
                    for key, result in temp_results.items():
                        loggers[key].add_result(i, result)

                msg = ''
                for key, result in temp_results.items():
                    valid_res, test_res = result
                    msg += (key + f'{{v: {valid_res:.4f}, t: {test_res:.4f}}} ')
                print('epoch :{:3d}, loss:{:.4f}, loss_v:{:.4f}, loss_t:{:.4f}'
                      .format(j, loss_k, temp_loss['v_loss'], temp_loss['t_loss']),
                      msg,
                      'avr_t: {:.1f}s'.format(np.mean(train_time)),
                      'T2: {:.1f}s  '.format(model.T2 / (j+1)),
                      )
                loss_val_j.append(temp_loss['v_loss'])
            if (j + 1) % 5 == 0:
                gc.collect()
            model.scheduler.step()

        print("lamda={}, lr={}, drop={}".format(str(args.lamda), str(args.lr), str(args.drop)))
        for key in loggers.keys():
            loggers[key].print_statistics(i, key=key)
        for key in loggers.keys():
            loggers[key].print_statistics(key=key, avr=i + 1)

        if args.save:
            import os.path as osp
            node_emb = model.get_node_embedding()
            path = osp.join(dataset.root, 'processed/GNN_emb.pt')
            joblib.dump(node_emb.numpy(), path, compress=3)
        del model
        gc.collect()

        sampler_tr.terminate()
        sampler_node.terminate()

    print("lamda={}, lr={}, drop={}".format(str(args.lamda), str(args.lr), str(args.drop)))
    for key in loggers.keys():
        loggers[key].print_statistics(key=key)
print(args)



