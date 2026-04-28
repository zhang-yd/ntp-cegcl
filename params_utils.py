import argparse
import sys

def set_params():
    # argv = sys.argv
    # dataset = argv[1]
    parser = argparse.ArgumentParser()
    
    ## basic info
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## model
    parser.add_argument('--k_hop', type=int, default=10)
    parser.add_argument('--cluster', type=float, default=0.09)
    parser.add_argument('--hidden_channels', type=int, default=1024)#1024
    parser.add_argument('--temperature', type=float, default=0.09)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4096)

    ## optimizer
    parser.add_argument('--epochs', type=int, default=15) #5
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--kd_epochs', type=int, default=500)
    parser.add_argument('--kd_lr', type=float, default=0.01)
    parser.add_argument('--kd_weight_decay', type=float, default=0.00005)

    parser.add_argument('--cls_epochs', type=int, default=600)
    parser.add_argument('--cls_lr', type=float, default=0.1)
    parser.add_argument('--cls_weight_decay', type=float, default=0.05)

    ## dataset type
    parser.add_argument('--big', type=bool, default=False)
    parser.add_argument('--homogeneous', type=bool, default=True)

    #####################################
    args, _ = parser.parse_known_args()
    return args

