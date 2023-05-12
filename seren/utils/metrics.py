import torch
import numpy as np
from scipy.stats import entropy

def accuracy_calculator(rank_list, last, kpis):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1).double().mean().item(),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]).mean().item(),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))]).mean().item()
    }

    
    return [metrics[kpi] for kpi in kpis]

#def diversity_calculator(rank_list, item_cate_matrix):
#    rank_list = rank_list.long()
#    ILD_perList = []
#    for b in range(rank_list.size(0)):
#        ILD = []
#        for i in range(len(rank_list[b])):
#            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
#            for j in range(i + 1, len(rank_list[b])):
#                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
#                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
#                ILD.append(distance)
#        ILD_perList.append(np.mean(ILD))
#
#
#    return torch.tensor(ILD_perList).mean().item()

#def diversity_calculator(rank_list, item_cate_matrix, kpis):
#    rank_list = rank_list.long()
#    ILD_perList = []
#    IE_perList = []
#    ds_perList = []
#    for b in range(rank_list.size(0)):
#        ILD = []
#        cate_dis = torch.Tensor([])
#        for i in range(len(rank_list[b])):
#            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
#            cate_dis = torch.cat((cate_dis, item_i_cate), 0)
#            for j in range(i + 1, len(rank_list[b])):
#                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
#                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
#                ILD.append(distance)
#        ILD_perList.append(np.mean(ILD))
#        cate_dis_sum = cate_dis.reshape(-1, item_cate_matrix.shape[1]).sum(axis=0)
#        IE_perList.append(entropy(np.array(cate_dis_sum/sum(cate_dis_sum)), base=2))
#        # pytorch 1.6 hasn't torch.count_nonzero(v1.10)
#        ds_perList.append(len(cate_dis_sum.nonzero()))
#
#    metrics = {
#        'ild': torch.tensor(ILD_perList).mean().item(),
#        'entropy': torch.tensor(IE_perList).mean().item(),
#        'diversity_score': torch.tensor(ds_perList).mean().item()/rank_list.size(1)
#    }
#    return [metrics[kpi] for kpi in kpis]

def diversity_calculator(rank_list, item_cate_matrix, kpis, lens=None):
    rank_list = rank_list.long()
    ILD_perList = []
    IE_perList = []
    ds_perList = []
    for b in range(rank_list.size(0)):
        ILD = []
        cate_dis = torch.Tensor([])
        sess_len = lens[b] if lens else len(rank_list[b])
        for i in range(sess_len):
            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
            cate_dis = torch.cat((cate_dis, item_i_cate), 0)
            for j in range(i + 1, sess_len):
                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
                ILD.append(distance)
        ILD_perList.append(np.mean(ILD))
        cate_dis_sum = cate_dis.reshape(-1, item_cate_matrix.shape[1]).sum(axis=0)
        IE_perList.append(entropy(np.array(cate_dis_sum/sum(cate_dis_sum)), base=2))
        # pytorch 1.6 hasn't torch.count_nonzero(v1.10)
        ds_perList.append(len(cate_dis_sum.nonzero())/sess_len)

    entropy_batch = torch.FloatTensor(IE_perList)
    entropy_batch[entropy_batch!=entropy_batch] = 0

    metrics = {
        'ild': torch.FloatTensor(ILD_perList).mean().item(),
        'entropy': entropy_batch.mean().item(),
        'diversity_score': torch.FloatTensor(ds_perList).mean().item()
    }
    return [metrics[kpi] for kpi in kpis]

def performance_calculator(rank_list, last, item_cate_matrix, acc_kpis, div_kpis, lens=None, if_f_score=True):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1).double().mean().item(),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]).mean().item(),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))]).mean().item()
    }

    rank_list = rank_list.long()
    ILD_perList = []
    IE_perList = []
    ds_perList = []
    for b in range(rank_list.size(0)):
        ILD = []
        cate_dis = torch.Tensor([])
        sess_len = lens[b] if lens else len(rank_list[b])
        for i in range(sess_len):
            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
            cate_dis = torch.cat((cate_dis, item_i_cate), 0)
            for j in range(i + 1, sess_len):
                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
                ILD.append(distance)
        ILD_perList.append(np.mean(ILD))
        cate_dis_sum = cate_dis.reshape(-1, item_cate_matrix.shape[1]).sum(axis=0)
        IE_perList.append(entropy(np.array(cate_dis_sum/sum(cate_dis_sum)), base=2))
        # pytorch 1.6 hasn't torch.count_nonzero(v1.10)
        ds_perList.append(len(cate_dis_sum.nonzero())/sess_len)

    entropy_batch = torch.FloatTensor(IE_perList)
    entropy_batch[entropy_batch!=entropy_batch] = 0
    
    div_metrics = {
        'ild': torch.FloatTensor(ILD_perList).mean().item(),
        'entropy': entropy_batch.mean().item(),
        'diversity_score': torch.FloatTensor(ds_perList).mean().item()
    }
    
    f_score = 0
    if if_f_score:
        hr_batch = hr.sum(axis=1)
        ild_batch = torch.FloatTensor(ILD_perList)
        f_score = 2*hr_batch*ild_batch/(hr_batch+ild_batch)
        f_score[torch.isnan(f_score)] = 0
    
    return [metrics[kpi] for kpi in acc_kpis], [div_metrics[kpi] for kpi in div_kpis], f_score.mean().item()


def performance_calculator_new(rank_list, last, item_cate_matrix, acc_kpis, div_kpis, aggregate_kpis, lens=None, if_f_score=True):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1).double().mean().item(),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]).mean().item(),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))]).mean().item()
    }

    rank_list = rank_list.long()
    ILD_perList = []
    IE_perList = []
    ds_perList = []
    for b in range(rank_list.size(0)):
        ILD = []
        cate_dis = torch.Tensor([])
        sess_len = lens[b] if lens else len(rank_list[b])
        for i in range(sess_len):
            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
            cate_dis = torch.cat((cate_dis, item_i_cate), 0)
            for j in range(i + 1, sess_len):
                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
                ILD.append(distance)
        ILD_perList.append(np.mean(ILD))
        cate_dis_sum = cate_dis.reshape(-1, item_cate_matrix.shape[1]).sum(axis=0)
        IE_perList.append(entropy(np.array(cate_dis_sum/sum(cate_dis_sum)), base=2))
        # pytorch 1.6 hasn't torch.count_nonzero(v1.10)
        ds_perList.append(len(cate_dis_sum.nonzero())/sess_len)
    
    entropy_batch = torch.FloatTensor(IE_perList)
    entropy_batch[entropy_batch!=entropy_batch] = 0
    
            
    div_metrics = {
        'ild': torch.FloatTensor(ILD_perList).mean().item(),
        'entropy': entropy_batch.mean().item(),
        'diversity_score': torch.FloatTensor(ds_perList).mean().item()
    }
    
    if if_f_score:
        beta = 0.5 # diversity is 0.5 important than accuracy
        
        hr_batch = hr.sum(axis=1)
        ild_batch = torch.FloatTensor(ILD_perList)
        f_score_hr_ild = 2 * hr_batch * ild_batch/(hr_batch + ild_batch)
        f_beta_score_hr_ild = (1+beta**2) * hr_batch * ild_batch/((beta**2)*hr_batch + ild_batch)
        f_score_hr_ild[torch.isnan(f_score_hr_ild)] = 0
        f_beta_score_hr_ild[torch.isnan(f_beta_score_hr_ild)] = 0
        
        ds_batch = torch.FloatTensor(ds_perList)
        f_score_hr_ds = 2 * hr_batch * ds_batch/(hr_batch + ds_batch)
        f_beta_score_hr_ds = (1+beta**2) * hr_batch * ds_batch/((beta**2)*hr_batch + ds_batch)
        f_score_hr_ds[torch.isnan(f_score_hr_ds)] = 0
        f_beta_score_hr_ds[torch.isnan(f_beta_score_hr_ds)] = 0
        
    aggregate_metrics = {
        'f_hr_ild': f_score_hr_ild.mean().item(),
        'f_beta_hr_ild': f_beta_score_hr_ild.mean().item(),
        'f_hr_ds': f_score_hr_ds.mean().item(),
        'f_beta_hr_ds': f_beta_score_hr_ds.mean().item()
    }
    
    
    return [metrics[kpi] for kpi in acc_kpis], [div_metrics[kpi] for kpi in div_kpis], [aggregate_metrics[kpi] for kpi in aggregate_kpis]
    
