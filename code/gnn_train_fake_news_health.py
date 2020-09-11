import torch
import torch.nn.functional as F
import argparse, time, datetime, shutil
import sys, os, glob, json, random
import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
# from torchsummary import summary

from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
import nltk
nltk.download('punkt')

# from models.gnn_model import Graph_Net, Relational_GNN
# from models.transformer_model import *
from utils.utils import *
from gnn_train.gnn_train_main import *
from caching_funcs.cache_gnn import *







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type = str, default = './data/complete_data',
                          help='path to dataset folder that contains the adj and feat matrices, etc')
    parser.add_argument('--model_checkpoint_path', type = str, default = './model_checkpoints_gnn',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type = str, default = './vis_checkpoints_gnn',
                          help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model_lr.pt',
                       help = 'saved model name')
    
    #### Training Params -- Euclidean GNNs ####
    
    # Named params    
    parser.add_argument('--data_name', type = str, default = 'gossipcop',
                          help='dataset name: politifact / gossipcop / pheme / rumoreval / HealthStory / HealthRelease')
    parser.add_argument('--model_name', type = str, default = 'HGCN',
                          help='model name: gcn / graph_sage / graph_conv / gat / rgcn / rsage / rgat / hgcn / HGCN / HNN')
    parser.add_argument('--saint', type = str, default = 'node',
                          help='which GraphSAINT sampling to use: random_walk / node / edge')
    parser.add_argument('--mode', type = str, default = 'lr',
                          help='Whether to train in transductive (normal) way or inductive way (lr)')
    parser.add_argument('--optimizer', type = str, default = 'RAdam',
                        help = 'Optimizer to use for training')
    parser.add_argument('--loss_func', type = str, default = 'bce_logits',
                        help = 'Loss function to use for optimization: bce / bce_logits / ce')
    parser.add_argument('--scheduler', type = str, default = 'step',
                        help = 'The type of lr scheduler to use anneal learning rate: step/multi_step')
    
    # Dimensions/sizes params   
    parser.add_argument('--batch_size', type = int, default = 16,
                          help='batch size for training"')
    parser.add_argument('--embed_dim', type = int, default = 16,
                          help='dimension of hidden layers of the graph network')
    parser.add_argument('--fc_dim', type = int, default = 64,
                          help='dimension of hidden layers of the MLP classifier')

    
    # Numerical params
    parser.add_argument('--num_rels', type = int, default = 3,
                          help='No. of types of edges present"')
    parser.add_argument('--clusters', type = int, default = 300,
                          help='No. of clusters of sub-graphs for cluster-GCN"')
    parser.add_argument('--pos_wt', type = float, default = 3,
                          help='Loss reweighting for the positive class to deal with class imbalance')
    parser.add_argument('--lr', type = float, default = 5e-3,
                          help='Learning rate for training')
    parser.add_argument('--weight_decay', type = float, default = 1e-3,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', type = float, default = 0.8,
                        help = 'Momentum for optimizer')
    parser.add_argument('--max_epoch', type = int, default = 100,
                        help = 'Max epochs to train for')
    parser.add_argument('--lr_decay_step', type = float, default = 5,
                        help = 'No. of epochs after which learning rate should be decreased')
    parser.add_argument('--lr_decay_factor', type = float, default = 0.8,
                        help = 'Decay the learning rate of the optimizer by this multiplicative amount')
    parser.add_argument('--patience', type = float, default = 10,
                        help = 'Patience no. of epochs for early stopping')
    parser.add_argument('--dropout', type = float, default = 0.2,
                        help = 'Regularization - dropout on hidden embeddings')
    parser.add_argument('--node_drop', type = float, default = 0.2,
                        help = 'Node dropout to drop entire node from a batch')
    parser.add_argument('--seed', type=int, default=21,
                        help='set seed for reproducability')
    parser.add_argument('--log_every', type=int, default=2000,
                        help='Log stats in Tensorboard every x iterations (not epochs) of training')
    
    # Options params
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle batches')
    parser.add_argument('--cluster', type=bool, default=True,
                        help='whether to apply graph clustering before batching (higher priority than SAINT sampling)')
    parser.add_argument('--full_graph', type=bool, default=False,
                        help='whether to process the entire graph without clustering or sampling')
    
    
    #### Training Params -- Hyperbolic GNN ####
    
    parser.add_argument('--train_task', type = str, default = 'nc',
                          help='Task for H-GCN: node classif (nc) or link prediction (lp)')
    parser.add_argument('--manifold', type = str, default = 'PoincareBall',
                          help='Manifold to use: Euclidean, Hyperboloid, PoincareBall')
    parser.add_argument('--c', type = int, default = None,
                          help='Hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--bias', type = int, default = 1,
                          help='Use bias term or not')
    parser.add_argument('--use_att', type = bool, default = False,
                          help='Use attention during aggregation or not: with attn = HyGAT, without attn = HyGCN')
    parser.add_argument('--pos_weight', type = int, default = None,
                          help='Werigh for positive class for loss re-weighting')
    

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device   

    if config['data_name'] == 'pheme':
        config['n_classes'] = 3
        config['loss_func'] = 'ce'
    else:
        config['n_classes'] = 1
        config['loss_func'] = 'bce_logits'
    
    args.model = config['model_name']
    args.num_layers = 2
    args.act = 'relu'
    args.dim = config['embed_dim']
    args.task = config['train_task']
    args.cuda = -1 if device == 'cpu' else device
   
    # Check all provided paths:    
    config['model_path'] = os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'])
    config['vis_path'] = os.path.join(config['vis_path'], config['data_name'], config['model_name'])
        
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("\nData path checked..")   
    if not os.path.exists(config['model_path']):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(config['model_path']))
        os.makedirs(config['model_path'])
    else:
        print("\nModel save path checked..")
    if config['model_name'] not in ['gcn', 'graph_sage', 'graph_conv', 'gat', 'rgcn', 'rsage', 'rgat', 'HGCN', 'HNN']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - gcn / graph_sage / graph_conv / gat / rgcn / rsage / rgat / HGCN / HNN")
    else:
        print("\nModel name checked...")
    if not os.path.exists(config['vis_path']):
        print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
        os.makedirs(config['vis_path'])
    else:
        print("\nTensorbaord Visualization path checked..")
        print("Cleaning Visualization path of older tensorboard files...\n")
        #shutil.rmtree(config['vis_path'])

    

    # Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)
    
    # # Prepare the tensorboard writer
    # writer = SummaryWriter(config['vis_path'])
    
    # Prepare dataset and iterators for training
    fold =1
    #config['loader'], config['vocab_size'], config['data'] = prepare_gnn_training(config, fold=fold)
    #del config['loader']
    #del config['vocab_size']
    #del config['data']
    
    # For FakeNews non-transfomer training - lr
    #seeds = [21, 42]
    #w_decay = [1e-3, 2e-3]
    #dropouts = [0.1, 0.2, 0.3, 0.4]
    #node_dropout = [0.1, 0.2, 0.3, 0.4]
    #lrs = [5e-3, 1e-4, 5e-4]
    #embed_dims = [128, 256, 512]
    seeds = [3, 21, 42, 84, 168]
    w_decay = [1e-3]
    dropouts = [0.1]
    node_dropout = [0.1]
    lrs = [5e-3]
    embed_dims = [256]
    
    best_drop, best_node_drop, best_decay, best_lr = 0,0,0,0
    best_f1= -100
    results = {}   
    
    for decay in w_decay:
        results[decay] = {}
        config['weight_decay'] = decay
        for lr in lrs:
            results[decay][lr] = {}
            config['lr'] = lr
            for hid_drop in dropouts:
                results[decay][lr][hid_drop] = {}
                config['dropout'] = hid_drop
                args.dropout = hid_drop
                for node_drop in node_dropout:
                    results[decay][lr][hid_drop][node_drop] = {}
                    config['node_drop'] = node_drop
                    args.node_drop = node_drop
                    for embed_dim in embed_dims:
                        results[decay][lr][hid_drop][node_drop][embed_dim] = {}
                        config['embed_dim'] = embed_dim
                        args.dim = embed_dim
                        avg_val_recall, avg_val_prec, avg_val_acc = 0,0,0
                        avg_val_f1 = 0
                        if hid_drop>0.2 and node_drop>0.2:
                            print("\nSKIPPINGGGG:  decay = {}, lr = {}, dropout = {}, node_drop = {}, embed_dim = {}".format(decay, lr, hid_drop, node_drop, embed_dim))
                            continue
                        for seed in seeds:
                            config['seed']= seed
                            print("\n decay = {}, lr = {}, dropout = {}, node_drop = {}, seed = {}, embed_dim = {}".format(decay, lr, hid_drop, node_drop, seed, embed_dim))
                            config['writer'] = SummaryWriter(config['vis_path'])
                            
                            # Seeds for reproduceable runs
                            torch.manual_seed(config['seed'])
                            torch.cuda.manual_seed(config['seed'])
                            np.random.seed(config['seed'])
                            random.seed(config['seed'])
                            torch.backends.cudnn.deterministic = True
                            torch.backends.cudnn.benchmark = False
                            
                            config['loader'], config['vocab_size'], config['data'] = prepare_gnn_training(config, fold=fold, verbose=False)
                            args.n_nodes, args.feat_dim = config['data'].x.shape
                            # sys.exit()
                            
                            graph_net = Graph_Net_Main(config, args)
                            best_val_f1, best_val_acc, best_val_recall, best_val_precision = graph_net.train_main(cache=True)
                            avg_val_f1 += best_val_f1
                            avg_val_prec+= best_val_precision
                            avg_val_recall+= best_val_recall
                            avg_val_acc+= best_val_acc
                            del config['loader']
                            del config['vocab_size']
                            del graph_net
                            torch.cuda.empty_cache()
                            gc.collect()
                            
        
                            config['writer'].close()
                        results[decay][lr][hid_drop][node_drop][embed_dim]['f1'] = avg_val_f1 /len(seeds)
                        results[decay][lr][hid_drop][node_drop][embed_dim]['precision'] = avg_val_prec/len(seeds)
                        results[decay][lr][hid_drop][node_drop][embed_dim]['recall'] = avg_val_recall/len(seeds)
                        results[decay][lr][hid_drop][node_drop][embed_dim]['accuracy'] = avg_val_acc/len(seeds)
                        
                        if results[decay][lr][hid_drop][node_drop][embed_dim]['f1'] > best_f1:
                            best_f1 = results[decay][lr][hid_drop][node_drop][embed_dim]['f1']
                            best_drop = hid_drop
                            best_node_drop = node_drop
                            best_decay = decay
                            best_lr = lr
                            best_embed_dim = embed_dim
                        print("\n\n", results)
                        print("Best so far: \nbest_f1 = {},  best_drop = {},  best_node_drop = {},  best_lr = {},  best_embed_dim = {},  best_decay = {}".format(best_f1, best_drop, best_node_drop, best_lr, best_embed_dim, best_decay))
                    
    print("\n\nModel with best f1 score has the configuration:\n\n")
    print("Weight_decay  =  ", best_decay)
    print("LR  =  ", best_lr)
    print("Dropout  =  ", best_drop)
    print("Node Dropout  =  ", best_node_drop)
    print("Embedding dim = ", best_embed_dim)
    print("\n\n..... with the scores:\n")
    print("Accuracy = ", results[best_decay][best_lr][best_drop][best_node_drop][best_embed_dim]['accuracy'])
    print("F1 = ", results[best_decay][best_lr][best_drop][best_node_drop][best_embed_dim]['f1'])
    print("Precision = ", results[best_decay][best_lr][best_drop][best_node_drop][best_embed_dim]['precision'])
    print("Recall = ", results[best_decay][best_lr][best_drop][best_node_drop][best_embed_dim]['recall'])
     
    
    # try:
    #     graph_net = Graph_Net_Main(config)
    #     graph_net.train_main()
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
    #     print("Best val f1 = ", graph_net.best_val_f1)
    #     writer.close()