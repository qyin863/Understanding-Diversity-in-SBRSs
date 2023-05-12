import logging

ACC_KPI = ['ndcg', 'mrr', 'hr']
DIV_KPI = ['ild', 'entropy', 'diversity_score']
AGG_KPI = ['f_hr_ild', 'f_beta_hr_ild', 'f_hr_ds', 'f_beta_hr_ds']

CAT_COLS = {
    'ml-100k': []
}

MAX_LEN = 20
DATA_PATH = './dataset/'
TUNE_PATH = './tune_log/'

def get_logger(file_name):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    log_file = file_name + '.log'

    fileHandler = logging.FileHandler(log_file, mode = 'w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # logger.info("test")

    return logger

def get_parameters(args):
    params = {
        'model': args.model,
        'user_key': args.user_key,
        'item_key': args.item_key,
        'category_key': args.category_key,
        'session_key': args.session_key,
        'time_key': args.time_key,
        # 'split_ratio': 0.8,
        'output_path': f'./temp/{args.dataset}/{args.desc}/',
        'description': args.desc,
        'dataset': args.dataset,
        'session_len': MAX_LEN,
        'topk': args.topk,
    }

    model_params = {
        'batch_size': args.batch_size,
        'item_embedding_dim': args.item_embedding_dim, # -1 for gru4rec means not use
        'hidden_size': args.hidden_size,
        'epochs': args.epochs,
        'sigma': args.sigma,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'l2': args.l2,
        'lr_dc_step': args.lr_dc_step,
        'lr_dc': args.lr_dc,
        'n_layers': args.n_layers,
        'step': args.step,
        'dropout_hidden': args.dropout_hidden,
        'dropout_input': args.dropout_input,
        'final_act': args.final_act,
        'loss_type': args.loss_type,
        'wd': args.weight_decay,
        'eps': args.eps,
        'momentum': args.momentum,
        'pop_n': args.pop_n,
        'n_sims': args.n_sims,
        'lambda': args.lmbd,
        'alpha': args.alpha,
        'lambda_item': args.lambda_item,
        'lambda_session': args.lambda_session,
        # for gcegnn
        'activate': args.activate,
        'n_sample_all': args.n_sample_all,
        'n_sample': args.n_sample,
        'n_iter': args.n_iter,
        'dropout_gcn': args.dropout_gcn,
        'dropout_local':args.dropout_local,
        'dropout_global': args.dropout_global,
        # for gcsan
        'weight': args.weight,
        'blocks': args.blocks,
        # for mcprn
        'tau': args.tau,
        'purposes': args.purposes,
    }

    return params, model_params
