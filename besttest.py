import torch
import argparse

from seren.utils.data import Interactions, Categories
from seren.config import get_parameters, get_logger, ACC_KPI, DIV_KPI
from seren.utils.model_selection import fold_out, train_test_split, handle_adj, build_graph
from seren.utils.dataset import NARMDataset, SRGNNDataset, GRU4RECDataset, ConventionDataset, GCEDataset
from seren.utils.metrics import accuracy_calculator, diversity_calculator, performance_calculator
from seren.model.narm import NARM
from seren.model.stamp import STAMP
from seren.model.mcprn_v4_block import MCPRN
from seren.model.srgnn import SessionGraph
from seren.model.gcsan_v2 import GCSAN
from seren.model.gcegnn import CombineGraph
from seren.model.gru4rec import GRU4REC
from seren.model.conventions import Pop, SessionPop, ItemKNN, BPRMF, FPMC
from seren.utils.functions import reindex
from seren.config import TUNE_PATH
from hyperopt import hp, tpe, fmin
import json
import os
import numpy as np

def best_func(param_dict, args, f):
    # specify certain parameter according to algo_name
    # common hyperparameters
    hypers_dict = dict()
    hypers_dict['item_embedding_dim'] = int(param_dict['item_embedding_dim']) if 'item_embedding_dim' in param_dict.keys() else args.item_embedding_dim
    hypers_dict['lr'] = param_dict['lr'] if 'lr' in param_dict.keys() else args.lr
    hypers_dict['batch_size'] = int(param_dict['batch_size']) if 'batch_size' in param_dict.keys() else args.batch_size
    hypers_dict['epochs'] = int(param_dict['epochs']) if 'epochs' in param_dict.keys() else args.epochs
    
    # narm&gru4rec addition
    hypers_dict['hidden_size'] = int(param_dict['hidden_size']) if 'hidden_size' in param_dict.keys() else args.hidden_size
    hypers_dict['n_layers'] = int(param_dict['n_layers']) if 'n_layers' in param_dict.keys() else args.n_layers
    
    # gru4rec addition
    hypers_dict['dropout_input'] = param_dict['dropout_input'] if 'dropout_input' in param_dict.keys() else args.dropout_input
    hypers_dict['dropout_hidden'] = param_dict['dropout_hidden'] if 'dropout_hidden' in param_dict.keys() else args.dropout_hidden
    
    # srgnn&gcsan addition
    hypers_dict['step'] = int(param_dict['step']) if 'step' in param_dict.keys() else args.step
    
    # itemknn addition
    hypers_dict['alpha'] = param_dict['alpha'] if 'alpha' in param_dict.keys() else args.alpha
    
    # gcegnn addition
    hypers_dict['n_iter'] = int(param_dict['n_iter']) if 'n_iter' in param_dict.keys() else args.n_iter # [1, 2]
    hypers_dict['dropout_gcn'] = param_dict['dropout_gcn'] if 'dropout_gcn' in param_dict.keys() else args.dropout_gcn # [0, 0.2, 0.4, 0.6, 0.8]
    hypers_dict['dropout_local'] = param_dict['dropout_local'] if 'dropout_local' in param_dict.keys() else args.dropout_local # [0, 0.5]


    hypers_dict['weight'] = param_dict['weight'] if 'weight' in param_dict.keys() else args.weight
    hypers_dict['blocks'] = int(param_dict['blocks']) if 'blocks' in param_dict.keys() else args.blocks
    hypers_dict['tau'] = param_dict['tau'] if 'tau' in param_dict.keys() else args.tau
    hypers_dict['purposes'] = param_dict['purposes'] if 'purposes' in param_dict.keys() else args.purposes
    
    
    # reset hyperparmeters in args using hyperopt param_dict value
    args.item_embedding_dim = 2 #hypers_dict['item_embedding_dim']
    args.lr = hypers_dict['lr']
    args.batch_size = hypers_dict['batch_size']
    args.epochs = hypers_dict['epochs']
    # narm&gru4rec addition
    args.hidden_size = hypers_dict['hidden_size']
    args.n_layers = hypers_dict['n_layers']
    # gru4rec addition
    args.dropout_input = hypers_dict['dropout_input']
    args.dropout_hidden = hypers_dict['dropout_hidden']
    # srgnn&gcsan addition
    args.step = hypers_dict['step']
    
    args.alpha = hypers_dict['alpha']
    
    # gcegnn addition
    args.n_iter = hypers_dict['n_iter']
    args.dropout_gcn = hypers_dict['dropout_gcn']
    args.dropout_local = hypers_dict['dropout_local']

    args.weight = hypers_dict['weight']
    args.blocks = hypers_dict['blocks']
    args.tau = hypers_dict['tau']
    args.purposes = hypers_dict['purposes']
    
#    print(args.alpha)
    # new conf and model_conf
    conf, model_conf = get_parameters(args)

    logger = get_logger(__file__.split('.')[0] + f'_{conf["description"]}')

    ds = Interactions(conf, logger)
    if conf['model'] in ['narm','fpmc','stamp','bprmf','mcprn']:
    #    train, test = fold_out(ds.df, conf)
    #    train, valid = fold_out(train, conf)
        train, test = train_test_split(ds.df, conf, logger, n_days=ds.n_days)#fold_out(ds.df, conf)
        train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=False)
        test = reindex(test, conf['item_key'], item_id_map) # code from 1
        item_num = train[conf['item_key']].nunique()
        train_dataset = NARMDataset(train, conf)
        #valid_dataset = NARMDataset(valid, conf)
        test_dataset = NARMDataset(test, conf)
        # logger.debug(ds.item_num)
        train_loader = train_dataset.get_loader(model_conf, shuffle=True)
        #valid_loader = valid_dataset.get_loader(model_conf, shuffle=False)
        if conf['model'] == 'mcprn':
            model_conf['batch_size'] = 1
        test_loader = test_dataset.get_loader(model_conf, shuffle=False)
        if conf['model'] == 'narm':
            model = NARM(ds.item_num, model_conf, logger)
        elif conf['model'] == 'fpmc':
            model = FPMC(ds.item_num, model_conf, logger)
        elif conf['model'] == 'stamp':
            model = STAMP(ds.item_num, model_conf, logger)
        elif conf['model'] == 'bprmf':
            model = BPRMF(ds.item_num, model_conf, logger)
        elif conf['model'] == 'mcprn':
            model = MCPRN(item_num, model_conf, logger)
        else:
            assert 1!=1, 'model name error'
        model.fit(train_loader)#, valid_loader)
        if args.model in ['narm','stamp']:
            torch.save(model.item_embedding.weight, TUNE_PATH + args.model +'/2d_forUniformity_withPerformance/'+ args.dataset+'_embs.pt')
        if args.model=='mcprn':
            torch.save(model.emb.weight, TUNE_PATH + args.model +'/2d_forUniformity_withPerformance/'+ args.dataset+'_embs.pt')
            
        preds, truth = model.predict(test_loader, conf['topk'])
        
    elif conf['model'] in ['srgnn','gcsan']:
    #    train, test = fold_out(ds.df, conf)
    #    train, valid = fold_out(train, conf)
        train, test = train_test_split(ds.df, conf, logger, n_days=ds.n_days)#fold_out(ds.df, conf)
        train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=False)
        test = reindex(test, conf['item_key'], item_id_map, start_from_zero=False) # code from 1
        train_dataset = SRGNNDataset(train, conf, shuffle=True)
        #valid_dataset = SRGNNDataset(valid, conf, shuffle=False)
        test_dataset = SRGNNDataset(test, conf, shuffle=False)
        model = SessionGraph(ds.item_num, model_conf, logger) if conf['model']=='srgnn' else GCSAN(ds.item_num, model_conf, logger)
        model.fit(train_dataset)#, valid_dataset)
        
        torch.save(model.embedding.weight, TUNE_PATH + args.model +'/2d_forUniformity_withPerformance/'+ args.dataset+'_embs.pt')
        
        preds, truth = model.predict(test_dataset, conf['topk'])
        
    elif conf['model'] == 'gcegnn':
        train, test = train_test_split(ds.df, conf, logger, n_days=ds.n_days)#fold_out(ds.df, conf)
        train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=False)
        test = reindex(test, conf['item_key'], item_id_map, start_from_zero=False) # code from 1
        # adj, num
        adj, num = build_graph(train, conf, model_conf)
        train_dataset = GCEDataset(train, conf)
        test_dataset = GCEDataset(test, conf)
        num_node = train[conf['item_key']].nunique() + 1
    #    print(conf)
    #    print(model_conf)
        adj, num = handle_adj(adj, num_node, model_conf['n_sample_all'], num)
        model = CombineGraph(model_conf, num_node, adj, num, logger)
        model.fit(train_dataset)#, valid_dataset)
        torch.save(model.embedding.weight, TUNE_PATH + args.model +'/2d_forUniformity_withPerformance/'+ args.dataset+'_embs.pt')
        
        preds, truth = model.predict(test_dataset, conf['topk'])
    elif conf['model'] == 'gru4rec':
        train, test = train_test_split(ds.df, conf, logger)#fold_out(ds.df, conf)
        train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=True)
        test = reindex(test, conf['item_key'], item_id_map, start_from_zero=True)
        suitable_batch = min(
            model_conf['batch_size'],
            train[conf['session_key']].nunique(),
            test[conf['session_key']].nunique() #, valid[conf['session_key']].nunique()
        )
        if suitable_batch < model_conf['batch_size']:
            logger.warning(
                f'Currrent batch size {model_conf["batch_size"]} is not suitable, the maximum tolerance for batch size is {suitable_batch}')
            model_conf['batch_size'] = suitable_batch

        train_loader = GRU4RECDataset(train, conf, model_conf['batch_size'])
        test_loader = GRU4RECDataset(test, conf, model_conf['batch_size'])
        model = GRU4REC(len(item_id_map), model_conf, logger)
        model.fit(train_loader)#, valid_loader)
        
        torch.save(model.look_up.weight, TUNE_PATH + args.model +'/2d_forUniformity_withPerformance/'+ args.dataset+'_embs.pt')
        
        preds, truth = model.predict(test_loader, conf['topk'])
        
    elif conf['model'] in ['pop','spop','itemknn']:
        train, test = train_test_split(ds.df, conf, logger)#fold_out(ds.df, conf)
        train, item_id_map, id_item_map = reindex(train, conf['item_key'], None, start_from_zero=False)
        test = reindex(test, conf['item_key'], item_id_map)
        test_dataset = ConventionDataset(test, conf)
        if conf['model'] == 'pop':
            model = Pop(conf, model_conf, logger)
        elif conf['model'] == 'spop':
            model = SessionPop(conf, model_conf, logger)
        elif conf['model'] == 'itemknn':
            model = ItemKNN(conf, model_conf, logger)
        else:
            assert 1!=1, 'model name error'
        model.fit(train)
        preds, truth = model.predict(test_dataset)
        
    else:
        logger.error('Invalid model name')
        raise ValueError('Invalid model name')


    logger.info(f"Finish predicting, start calculating {conf['model']}'s KPI...")
#    metrics = accuracy_calculator(preds, truth, ACC_KPI)
    cats = Categories(item_id_map, conf, logger)
    
    torch.save(cats.item_cate_matrix, 'tune_log/'+conf['model']+'/2d_forUniformity_withPerformance/'+conf['dataset']+'_item_cate_matrix.pt')
    
#    metrics_div = diversity_calculator(preds, cats.item_cate_matrix, DIV_KPI)

    metrics, metrics_div, f_score = performance_calculator(preds[:,:20], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI)
    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
    f.write(str(args.model)+','+str(args.dataset)+', top20, '+','.join(foo) +','+','.join(div_foo)+','+str(f_score) + '\n')

    metrics, metrics_div, f_score = performance_calculator(preds[:,:10], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI)
    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
    f.write(str(args.model)+','+str(args.dataset)+', top10, '+','.join(foo) +','+','.join(div_foo)+','+str(f_score) + '\n')

    metrics, metrics_div, f_score = performance_calculator(preds[:, :5], truth, cats.item_cate_matrix, ACC_KPI, DIV_KPI)
    foo = ['%.4f'%(metrics[i]) for i in range(len(ACC_KPI))]
    div_foo = ['%.4f'%(metrics_div[i]) for i in range(len(DIV_KPI))]
    f.write(str(args.model)+','+str(args.dataset)+', top5, '+','.join(foo) +','+','.join(div_foo)+','+str(f_score) + '\n')
    f.flush()

    return None

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gru4rec", type=str)
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--category_key", default="category_id", type=str)
    parser.add_argument("--session_key", default="session_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)
    parser.add_argument("--dataset", default="ml-100k", type=str)
    parser.add_argument("--desc", default="nothing", type=str)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting

    parser.add_argument('--batch_size', type=int, default=128, help='batch size for loader')
    parser.add_argument('--item_embedding_dim', type=int, default=100, help='dimension of item embedding')
    parser.add_argument('--hidden_size', type=int, default=100, help='dimension of linear layer')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2/BPR penalty')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of gru layers')
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument("--sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature
    parser.add_argument('--dropout_input', default=0, type=float) #0.5 for TOP and 0.3 for BPR
    parser.add_argument('--dropout_hidden', default=0, type=float) #0.5 for TOP and 0.3 for BPR
    parser.add_argument('--optimizer', default='Adagrad', type=str)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--eps', default=1e-6, type=float) #not used
    parser.add_argument('--final_act', default='tanh', type=str)
    parser.add_argument('--loss_type', default='BPR-max', type=str) #type of loss function TOP1 / BPR for GRU4REC, TOP1-max / BPR-max for GRU4REC+
    parser.add_argument('--pop_n', type=int, default=100, help='top popular N items')
    parser.add_argument('--n_sims', type=int, default=100, help='non-zero scores to the N most similar items given back')
    parser.add_argument('--lmbd', type=float, default=20, help='Regularization. Discounts the similarity of rare items')
    parser.add_argument('--alpha', type=float, default=0.5, help='Balance between normalizing with the supports of the two items')
    parser.add_argument('--lambda_session', type=float, default=0, help='session embedding penalty')
    parser.add_argument('--lambda_item', type=float, default=0, help='item embedding penalty')
    # add for gcegnn
    parser.add_argument('--activate', type=str, default='relu')
    parser.add_argument('--n_sample_all', type=int, default=12)
    parser.add_argument('--n_sample', type=int, default=12)
    parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
    parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
    parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
    parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
    # add for gcsan
    parser.add_argument('--weight', type=float, default=0.4, help='weight in final session embedding')     # [0.4, 0.8]
    parser.add_argument('--blocks', type=int, default=1)                                    # [1,2,3,4]

    # add for mcprn
    parser.add_argument('--tau', type=float, default=0.1, help='tau in softmax')     # [0.4, 0.8]
    parser.add_argument('--purposes', type=int, default=1, help='#purposes in mcprn') 
    args = parser.parse_args()

    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    # begin tuning here
    tune_log_path = TUNE_PATH
    if not os.path.exists(tune_log_path+args.model):
        os.makedirs(tune_log_path+args.model)
    param_dict = json.load(open(tune_log_path + f'hypers/'+ args.model +'_'+ args.dataset+'.json', 'r'))
#        print(param_dict)
    acc_names = ['%s'%key for key in ACC_KPI]
    div_names = ['%s'%key for key in DIV_KPI]
#    f = open(tune_log_path+args.model +'/'+str(args.dataset)+'_resultwithBestHypers.txt','a',encoding='utf-8')
    f = open(tune_log_path+args.model +'/2d_forUniformity_withPerformance/performance.txt','a',encoding='utf-8')
    f.write('model,dataset,'+','.join(acc_names)+','+','.join(div_names)+',f-score'+'\n')
    f.flush()
    
#    best = fmin(opt_func, param_dict, algo=tpe.suggest, max_evals=2) #20
    for i in range(1):
        best_func(param_dict, args, f)
        
    f.close()
