import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import normal_, xavier_normal_
import torch.nn.functional as F

class Pop(object):
    def __init__(self, conf, params, logger):
        '''

        Parameters
        ----------
        pop_n : int
            Only give back non-zero scores to the top N ranking items. 
            Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        '''        
        self.top_n = params['pop_n']
        self.item_key = conf['item_key']
        self.session_key = conf['session_key']
        self.logger = logger

    def fit(self, train, valid_loader=None):
        grp = train.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)

    def predict(self, test, k=15):
        preds, last_item = torch.LongTensor([]), torch.LongTensor([])
        for seq, target, _ in test:
            pred = torch.LongTensor([self.pop_list.index[:k].tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item

class SessionPop(object):
    def __init__(self, conf, params, logger):
        '''
        Session popularity predictor that gives higher scores to items with higher number of occurrences in the session.
        
        Ties are broken up by adding the popularity score of the item.

        Parameters
        ----------
        pop_n : int
            Only give back non-zero scores to the top N ranking items.
            Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        '''
        self.top_n = params['pop_n']
        self.item_key = conf['item_key']
        self.session_key = conf['session_key']
        self.logger = logger

    def fit(self, train, valid_loader=None):
        grp = train.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        # get item_number for predict
        self.item_number = len(grp)

    def predict(self, test, k=15):
        pred_matrix, last_item = torch.LongTensor([]), torch.LongTensor([])
        for seq, target, _ in test:
            # version 1
            predict_for_item_ids = np.arange(self.item_number)
            preds = np.zeros(self.item_number)
            mask = np.in1d(predict_for_item_ids, self.pop_list.index)
            preds[mask] = self.pop_list[predict_for_item_ids[mask]]
            seq_items, item_counts = np.unique(seq, return_counts=True)
            ser = pd.Series(data=item_counts, index=seq_items)
            mask = np.in1d(predict_for_item_ids, ser.index)
            preds[mask] += ser[predict_for_item_ids[mask]]
            preds_series = pd.Series(data=preds, index=predict_for_item_ids)
            preds_series.sort_values(ascending=False, inplace=True)
            pred = torch.LongTensor([preds_series.index[:k].tolist()])
            pred_matrix = torch.cat((pred_matrix, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)
            
            # version 2
#            seq_items, item_counts = np.unique(seq, return_counts=True)
#            sers = pd.Series(data=seq_items, index=item_counts)
#            temp = pd.concat([self.pop_list, sers], axis=1).fillna(0)
#            temp.columns = [0, 1]
#            pop_list_current = temp[0] + temp[1]
#            pop_list_current.sort_values(ascending=False, inplace=True)
#            pred = torch.LongTensor([pop_list_current.index[:k].tolist()])
#            pred_matrix = torch.cat((pred_matrix, pred), 0)
#            last_item = torch.cat((last_item, torch.tensor(target)), 0)
            
        return pred_matrix, last_item
        
class ItemKNN(object):
    def __init__(self, conf, params, logger):
        '''        
        Item-to-item predictor that computes the the similarity to all items to the given item.
        
        Similarity of two items is given by:
        
        .. math::
            s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
            
        Parameters
        --------
        n_sims : int
            Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        lambda : float
            Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
        alpha : float
            Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
        '''    
        self.n_sims = params['n_sims']
        self.lmbd = params['lambda']
        self.alpha = params['alpha']
        self.item_key = conf['item_key']
        self.session_key = conf['session_key']
        self.time_key = conf['time_key']
        self.logger = logger

    def fit(self, data):
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids) 
        data = pd.merge(
            data, 
            pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}), 
            on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(
            data, 
            pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}), 
            on=self.session_key, how='inner')
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i+1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])

    def predict(self, test, k=15):
        preds, last_item = torch.LongTensor([]), torch.LongTensor([])

        for seq, target, _ in test:
            # cands_idx = ~np.in1d(self.sims[seq[-1]].index, seq)
            # pred = torch.tensor([self.sims[seq[-1]].index[cands_idx][:k].tolist()])
            pred = torch.LongTensor([self.sims[seq[-1]].index[:k].tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item

class BPRMF_1(object):
    def __init__(self, conf, params, logger, init_normal=True):
        '''
        BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
        
        Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of the session is modelled as the average of the feature vectors of the items that have occurred in it so far.
            
        Parameters
        --------
        n_factor : int
            The number of features in a feature vector. (Default value: 100)
        n_iterations : int
            The number of epoch for training. (Default value: 10)
        learning_rate : float
            Learning rate. (Default value: 0.01)
        lambda_session : float
            Regularization for session features. (Default value: 0.0)
        lambda_item : float
            Regularization for item features. (Default value: 0.0)
        sigma : float
            The width of the initialization. (Default value: 0.05)
        init_normal : boolean
            Whether to use uniform or normal distribution based initialization.
        session_key : string
            header of the session ID column in the input file (default: 'SessionId')
        item_key : string
            header of the item ID column in the input file (default: 'ItemId')
        '''
        self.n_factors = params['item_embedding_dim']
        self.n_iterations = params['epochs']
        self.learning_rate = params['learning_rate']
        self.lambda_session = params['lambda_session']
        self.lambda_item = params['lambda_item']
        self.sigma = params['sigma'] if params['sigma'] is not None else 0.05
        self.session_key = conf['session_key']
        self.item_key = conf['item_key']
        self.logger = logger

        self.init_normal = init_normal

    def init(self):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)

    def fit(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)

        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)

        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init()

        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            self.logger.info(f'training epoch: {it + 1}\tTrain Loss: {np.mean(c):.3f}')
            
    
    def predict(self, test, k=15):
        preds, last_item = torch.LongTensor([]), torch.LongTensor([])
        self.iditemmap = pd.Series(data=self.itemidmap.index, index=self.itemidmap.values)

        for seq, target, _ in test:
            iidx = self.itemidmap[seq].values
            uF = self.I[iidx].mean(axis=0)
            pred_iidx = np.argsort(self.I.dot(uF) + self.bI)[::-1]
            # cands_idx = ~np.in1d(pred_iidx, iidx)
            # pred_iidx = pred_iidx[cands_idx][:k]
            pred_iidx = pred_iidx[:k]
            pred = torch.LongTensor([self.iditemmap[pred_iidx].values.tolist()])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item

class BPRMF2(nn.Module):
    def __init__(self, item_num, conf, params, logger, init_normal=True):
        super(BPRMF2, self).__init__()
        self.n_factors = params['item_embedding_dim']
        self.n_iterations = params['epochs']
        self.learning_rate = params['learning_rate']
        self.lambda_item = params['lambda_item']
        self.sigma = params['sigma'] if params['sigma'] is not None else 0.05
        self.session_key = conf['session_key']
        self.item_key = conf['item_key']
        self.logger = logger

        self.init_normal = init_normal

        self.item_num = item_num
        self.embed_item = nn.Embedding(item_num, self.n_factors)
        if init_normal:
            nn.init.normal_(self.embed_item.weight, std=self.sigma)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, target, is_train=True):
        uF = self.embed_item(seq).mean(dim=0)
        iF = self.embed_item(target)
        if is_train:
            j = torch.randint(0, self.item_num - 1, (1, )) # sample 1
            while j.item() in target:
                j = torch.randint(0, self.item_num - 1, (1, ))
            j.to(self.device)
            jF = self.embed_item(j)

            pred_i = (uF * iF).sum(dim=-1)
            pred_j = (uF * jF).sum(dim=-1)

            return pred_i, pred_j
        else:
            pred_i = (uF * iF).sum(dim=-1)
            return pred_i

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.n_iterations + 1):
            self.train()
            total_loss, cnt = 0., 0
            for seq, target, _ in train_loader:
                seq = torch.tensor(seq).to(self.device)
                target = torch.tensor(target).to(self.device)

                self.optimizer.zero_grad()
                pred_i, pred_j = self.forward(seq, target)

                loss = -(pred_i - pred_j).sigmoid().log().sum() # BPR loss
                loss += self.lambda_item * self.embed_item.weight.norm()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                cnt += 1

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {total_loss / cnt:.3f}')
                
    def predict(self, test_loader, k=15):
        preds, last_item = torch.LongTensor([]).to(self.device), torch.LongTensor([]).to(self.device)
        self.eval()

        for seq, target, _ in test_loader:
            seq = torch.tensor(seq).to(self.device)
            target = torch.tensor(target).to(self.device)
            pred_is= self.forward(seq, torch.arange(self.item_num), is_train=False)
            _, rank_list = torch.topk(pred_is, k, -1)
            rank_list = torch.reshape(rank_list, (-1, k))
            preds = torch.cat((preds, rank_list), 0)
            last_item = torch.cat((last_item, target), 0)

        return preds.cpu(), last_item.cpu()


class FPMC(nn.Module):
    def __init__(self, n_items, params, logger):
        super(FPMC, self).__init__()
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.embedding_size = params['item_embedding_dim']
        self.n_items = n_items + 1
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.n_items, self.embedding_size)
        self.loss_func = BPRLoss()
        self.logger = logger
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
            
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_last_click_index = (torch.LongTensor(lengths)-1).to(self.device)
        item_last_click = torch.gather(seq, dim=1, index=item_last_click_index.unsqueeze(1)) # [b, 1]
        item_seq_emb = self.LI_emb(item_last_click.squeeze())  # [b,emb]
        
        #il_emb = self.IL_emb(next_item) # TODO next_item
        #il_emb = torch.unsqueeze(il_emb, dim=1)  # [b,n,emb] in here n = 1
        il_emb = self.IL_emb(torch.arange(self.n_items).to(self.device)) # [b,n,emb] in here n = item_size

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF  # MF part is dropped here because of anonymous user
#        mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))
#        mf = torch.squeeze(mf, dim=1)  # [B,1]
        #  FMC
        fmc = torch.matmul(item_seq_emb, il_emb.permute(1, 0)) # [b,n]
        #fmc = torch.squeeze(fmc, dim=1)  # [B,n]
        score = fmc #mf + fmc
        #score = torch.squeeze(score)
        return score
        
    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)
        self.logger.info('Start training...')
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []

            for _, (seq, target, lens) in enumerate(train_loader):
                seq = seq.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                logit = self.forward(seq, lens)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                total_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            s = ''
            if valid_loader:
                self.eval()
                val_loss = []
                with torch.no_grad():
                    for _, (seq, target, lens) in enumerate(valid_loader):
                        seq = seq.to(self.device)
                        target = target.to(self.device)
                        logit = self.forward(seq, lens)
                        logit_sampled = logit[:, target.view(-1)]
                        loss = self.loss_func(logit_sampled)
                        val_loss.append(loss.item())
                s = f'\tValidation Loss: {np.mean(val_loss):3f}'

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)
            
    def predict(self, test_loader, k=15):
        self.eval()
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, target_item, lens) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items

            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


class BPRMF(nn.Module):
    def __init__(self, item_num, params, logger):
        super(BPRMF, self).__init__()
        self.n_factors = params['item_embedding_dim']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.logger = logger

        self.n_items = item_num + 1
        self.item_embedding = nn.Embedding(self.n_items, self.n_factors, padding_idx=0)
        self.loss_func = BPRLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])
        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_seq_emb = self.item_embedding(seq)
        org_memory = item_seq_emb
        uF = torch.div(torch.sum(org_memory, dim=1), torch.FloatTensor(lengths).unsqueeze(1).to(self.device)) # [b, emb]
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(uF, item_embs.permute(1, 0))
        #item_scores = self.sf(scores)
        return scores


    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)
        self.logger.info('Start training...')
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []

            for _, (seq, target, lens) in enumerate(train_loader):
                seq = seq.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                logit = self.forward(seq, lens)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                total_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            s = ''
            if valid_loader:
                self.eval()
                val_loss = []
                with torch.no_grad():
                    for _, (seq, target, lens) in enumerate(valid_loader):
                        seq = seq.to(self.device)
                        target = target.to(self.device)
                        logit = self.forward(seq, lens)
                        logit_sampled = logit[:, target.view(-1)]
                        loss = self.loss_func(logit_sampled)
                        val_loss.append(loss.item())
                s = f'\tValidation Loss: {np.mean(val_loss):3f}'

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)
            
    def predict(self, test_loader, k=15):
        self.eval()
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, target_item, lens) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items

            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item
