import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from logger import SubgraphLogger

parser = argparse.ArgumentParser(description="Parser for KUCNet")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=str, default=42)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    torch.cuda.set_device(args.gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users   
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    if dataset == 'new_alibaba-fashion':  
        opts.lr = 0.00005
        opts.decay_rate = 0.999
        opts.lamb = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 50
    elif dataset == 'alibaba-fashion'  :
        opts.lr = 10**-6.5
        opts.decay_rate = 0.998
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2
        opts.act = 'relu'
        opts.n_batch = 10
        opts.n_tbatch = 10
        opts.K = 70
    elif dataset in ['last-fm', 'last-fm-reproduce', 'last-fm-subset']:
        opts.lr = 0.0004
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 30
        opts.n_tbatch = 30
        opts.K = 35
    elif dataset == 'new_last-fm' :
        opts.lr = 0.0004
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 36
        opts.n_tbatch = 36
        opts.K = 50
    elif dataset == 'new_amazon-book':
        opts.lr = 0.0005
        opts.decay_rate = 0.994  
        opts.lamb = 0.000014      
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 24
        opts.n_tbatch = 24
        opts.K = 170
    elif dataset in ['amazon-book', 'amazon-book-reproduce']:
        opts.lr = 0.0012
        opts.decay_rate = 0.994  
        opts.lamb = 0.000014      
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 120
    elif dataset == 'Dis_5fold_item'   :
        opts.lr = 0.0005
        opts.decay_rate = 0.994
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 35
    elif dataset == 'Dis_5fold_user'   :
        opts.lr = 0.001
        opts.decay_rate = 0.994
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 24
        opts.n_tbatch = 24
        opts.K = 550
    else:
        opts.lr = 0.0002
        opts.decay_rate = 0.9938
        opts.lamb = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = args.K


    config_str = '%d,%.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.K,opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    # Initialize logger
    logger = SubgraphLogger(results_dir=results_dir, dataset_name=dataset)

    best_recall = 0
    for epoch in range(15):
    
        print('epoch ',epoch)
        recall,ndcg, out_str = model.train_batch(logger)
        
        # --- LOGGING ---
        # Compute epoch summary
        summary = logger.compute_epoch_summary(
            n_params=model.n_params,
            train_time=model.t_time,
            inference_time=model.i_time,
            recall=recall,
            ndcg=ndcg
        )
        
        # Save epoch log
        log_file = logger.save_epoch_log(epoch, summary)
        print(f"[INFO] Saved epoch {epoch} log to {log_file}")
        
        # Print formatted summary
        print(logger.format_summary_string(summary))
        
        # Save to text file
        with open(opts.perf_file, 'a+') as f:
            f.write(f"Epoch {epoch}\n")
            f.write(logger.format_summary_string(summary))
            f.write("\n")
        # ------------------
        
        with open(opts.perf_file, 'a+') as f:
            f.write(str(epoch) + out_str)

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            best_summary = summary
            print("[BEST]" + str(epoch) + '\t' + best_str)
            
    # Save best model summary
    if best_summary is not None:
        best_log_file = logger.save_best_model_log(best_summary)
        print(f"[INFO] Saved best model summary to {best_log_file}")
    
    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n' + best_str)

    print(best_str)
    print(f"\n[INFO] All logs saved to {logger.logs_dir}/")