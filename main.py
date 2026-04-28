
import copy
import os
import random
import time

import numpy as np
import torch

import json
from datetime import datetime

from modules.logreg import LogReg

from params_utils import set_params
from model_new import MyGCL
from utils.dataset import load_dataset, load_large_dataset 
from utils.data_utils import eval_acc, class_rand_splits, load_fixed_splits, rand_train_test_idx 



def fix_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_result_json(dataset_name, data):
    now = datetime.now()
    now_time = now.strftime("%Y_%m_%d_%H_%M_%S") 

    json_file = f'result_{dataset_name}_{now_time}.json'
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_one_exp(args):
    fix_seed(args.seed)

    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    ### Load and preprocess data ###
    if args.big:
        dataset = load_large_dataset(args)
    else:
        dataset = load_dataset(args)

    if args.rand_split:
        split_idx_lst = [rand_train_test_idx(label=dataset.y, train_prop=args.train_ratio, valid_prop=args.valid_ratio)
                         for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.y, args.label_num_per_class, args.valid_num, args.test_num) for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(name=args.dataset_name)

    dataset = dataset.to(device)
    dataset.graph = dataset.graph.to(device)

    accs = []
    times = 0

    model = MyGCL(dataset.x.size(1), args.hidden_channels, args.alpha, args.temperature, args.dropout,
                    args.layer_norm, args.batch_norm, args.use_global_topology, args.global_topo_weight,
                    args.global_topo_decay, args.global_topo_hops).to(device)
    gcl_optimizer = torch.optim.Adam(model.gcl_model.parameters(), weight_decay=args.weight_decay, lr=args.lr) 

    rsd_criterion = torch.nn.SmoothL1Loss()
    cls_criterion = torch.nn.NLLLoss()

    if args.homogeneous:
        model.gcl_model._combination = True

    if args.big:
        model.gcl_model._combination = True

    model.train()

    ## Training GCL Model
    time_start = time.time()

    for epoch in range(int(args.epochs) + 1):
        gcl_optimizer.zero_grad()
        loss = model.gcl_model(dataset.x, dataset.node_to_par, dataset.P, dataset.A_P, dataset.graph)
        loss.backward()
        gcl_optimizer.step()
        print("loss ====== ", loss)

    time_end = time.time()
    times = time_end - time_start

    model.gcl_model.eval()
    dataset.graph = dataset.graph.to(device)

    with torch.no_grad():
        emb = model.adj_embed_add(dataset.x, dataset.graph, args.k_hop).to(device)

    print('Start testing...')
    model.eval()
    for i in range(10):
        split_idx = split_idx_lst[i]
        train_idx = split_idx['train'].to(device)
        valid_idx = split_idx['valid'].to(device)
        test_idx = split_idx['test'].to(device)

        best_acc_val = 0
        best_loss_val = 1e9
        final_test = 0
        logreg = LogReg(args.hidden_channels, dataset.y.max()+1).to(device)
        optimizer = torch.optim.Adam(logreg.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay)

        for _ in range(args.cls_epochs):
            logreg.train()
            optimizer.zero_grad()
            prob_train = torch.nn.functional.log_softmax(logreg(emb[train_idx]), dim=1)
            loss_cls = cls_criterion(prob_train, dataset.y[train_idx])
            loss_cls.backward()
            optimizer.step()

            logreg.eval()
            prob = torch.nn.functional.log_softmax(logreg(emb), dim=1)
            loss_val = torch.nn.functional.nll_loss(prob[valid_idx], dataset.y[valid_idx])
            acc_val = eval_acc(prob[valid_idx], dataset.y[valid_idx])
            acc_test = eval_acc(prob[test_idx], dataset.y[test_idx])

            if acc_val >= best_acc_val and best_loss_val >= loss_val:
                #print("better classification!")
                best_acc_val = max(acc_val, best_acc_val)
                best_loss_val = loss_val
                final_test = max(acc_test, final_test)

        accs.append(final_test.item())
        print(f'Run: {i:02d}, ' f'Test Accuracy: {final_test*100:.2f}')

    print(f'Test Accuracy: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}')
    print(f'Time per Epoch: {times/args.epochs:.4f}s, ' f'Total Time: {times:.4f}s')
    return accs


def main_func():
    args = set_params()
    print(args)

    topo_weight_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    topo_decay_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    topo_hops = [2, 3, 4]

    same_tick = 5

    data = {}
    for i in topo_weight_list:
        for j in topo_decay_list:
            for k in topo_hops:
                for a in range(1, same_tick + 1):
                    args.use_global_topology=True
                    args.global_topo_weight=i
                    args.global_topo_decay=j
                    args.global_topo_hops=k

                    accs = run_one_exp(args)
                    
                    accs_mean = np.mean(accs) 
                    accs_std = np.std(accs)
                    key = f'{i}_{j}_{k}_{a}'

                    data[key] = {
                        'topo_weight_list': i, 
                        'topo_decay_list': j, 
                        'topo_hops': k, 
                        'same_tick': a, 
                        'mean': accs_mean,
                        'std': accs_std,
                        'list': accs,
                    }

    write_result_json(args.dataset_name, data)


if __name__ == '__main__':
    print("=== start exp ====")
    main_func()
