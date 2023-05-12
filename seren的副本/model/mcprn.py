import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

epsilon = 1e-4
class PSRUCell(nn.Module):

    """
    An implementation of PSRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=False):
        super(PSRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden, concen):
        
        x = x.view(-1, x.size(1)) # -1 denotes batch_size
        
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        eps = 0.01
        multi = ((concen>=eps).int() * concen).unsqueeze(1) * inputgate
        
        hy = hidden + multi * (newgate - hidden)

        return hy
        
class MCPRN(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        embedding_dim(int): the dimension of item embedding
        purposes(int): the number of latent purposes

        hidden_size(int): the hidden size of gru
        
        batch_size(int):
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, params, logger): #, n_layers = 1
        super(MCPRN, self).__init__()
        self.n_items = n_items + 1 # 0 for None, so + 1
        self.epochs = params['epochs']
        self.embedding_dim = params['item_embedding_dim']
        self.purposes = 3
        self.logger = logger
        self.hidden_size = params['item_embedding_dim']#embedding_dim #hidden_size
        
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0) # With padding_idx set, the embedding vector at padding_idx is initialized to all zeros.
        #self.emb_dropout = nn.Dropout(0.25)
        self.embPurpose = nn.Embedding(self.purposes, self.embedding_dim)

        # 3-channels
        self.psru = PSRUCell(self.embedding_dim, self.hidden_size)#, self.n_layers)
        self.psru1 = PSRUCell(self.embedding_dim, self.hidden_size)
        self.psru2 = PSRUCell(self.embedding_dim, self.hidden_size)

        self.soft =nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # False

    def forward(self, seq, lens, tau=0.1 ):
        seq = seq.transpose(0, 1) # batch_size * seq_length
        # Purpose Router
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        attr_embs = self.embPurpose(torch.arange(self.purposes).to(self.device))
        item_embs = F.normalize(item_embs,p=2, dim=1)
        attr_embs = F.normalize(attr_embs, p=2, dim=1)


        gmatrix = torch.matmul(item_embs, attr_embs.t()) # |V|*3
        gmatrix = torch.exp(gmatrix/tau)
        gmatrix = gmatrix/gmatrix.sum(axis=1, keepdims=True)
        gmatrix[0] = 0 # index 0 pad item

        embs = self.emb(seq) #self.emb_dropout(self.emb(seq)) # batch_size * seq_len * embedding_dim
        #embs = pack_padded_sequence(embs, lengths)
        batch_size = seq.size(0)
        hidden = self.init_hidden(batch_size)
        
        hn = hidden
        hn1 = hidden
        hn2 = hidden
        
        gmat = gmatrix[:,0].unsqueeze(0).expand(batch_size, -1)
        gmat1 = gmatrix[:,1].unsqueeze(0).expand(batch_size, -1)
        gmat2 = gmatrix[:,2].unsqueeze(0).expand(batch_size, -1) # batch_size * |V|

        gmat_seq = torch.gather(gmat, 1, seq) # batch_size * |V|
        gmat1_seq = torch.gather(gmat1, 1, seq)
        gmat2_seq = torch.gather(gmat2, 1, seq)
        
        seq_len = seq.size(1)
        outs = torch.FloatTensor([]).to(self.device)
        for q in range(seq_len):  # seq.size(1)  seq_len
            hn = self.psru(embs[:,q,:], hn, gmat_seq[:,q]) # batch_size * embedding_dim
            hn1 = self.psru1(embs[:,q,:], hn1, gmat1_seq[:,q])
            hn2 = self.psru2(embs[:,q,:], hn2, gmat2_seq[:,q])
            #outs.append([hn,hn1,hn2])
            outs = torch.cat((outs, torch.cat((hn, hn1, hn2), 1).unsqueeze(0)), 0)
        # outs seq_len * batch_size * 3embedding_dim
        gmatExp, gmat1Exp, gmat2Exp = gmat.t().unsqueeze(2), gmat1.t().unsqueeze(2), gmat2.t().unsqueeze(2) #  |V| * batch_size * 1
        '''
        psru_outs = outs[-1] # [hn,hn1,hn2]  print(psru_outs.shape).  hn should batch_size * K
        hnExp, hn1Exp, hn2Exp = psru_outs[0].unsqueeze(0).expand(self.n_items,-1,-1), psru_outs[1].unsqueeze(0).expand(self.n_items,-1,-1), psru_outs[2].unsqueeze(0).expand(self.n_items,-1,-1) #  |V| * batch_size * K
        '''
        psru_outs = outs[torch.LongTensor(np.array(lens)-1).to(self.device), torch.arange(batch_size).to(self.device),:].squeeze() # batch_size * 3embedding_dim
        assert psru_outs.shape == (batch_size, 3*self.embedding_dim)
        psru_outs_hn, psru_outs_hn1, psru_outs_hn2 = psru_outs.chunk(3,1)
        hnExp = psru_outs_hn.expand(self.n_items, -1, -1)
        hn1Exp = psru_outs_hn1.expand(self.n_items, -1, -1)
        hn2Exp = psru_outs_hn2.expand(self.n_items, -1, -1)
        
        VC_cand_tensor = gmatExp * hnExp + gmat1Exp * hn1Exp + gmat2Exp * hn2Exp # |V| * batch_size * K

        # VC_cand_tensor = torch.cat(VC_cand)   # should |V| * batch_size * K
        VC_cand_tensor = VC_cand_tensor.permute(1,0,2) # should batch_size * |V| * K

        delta_c = VC_cand_tensor * item_embs.unsqueeze(0) # batch_size * |V| * K
        delta_c = delta_c.sum(2) # batch_size * |V|
        
        scores = self.sig(delta_c)
        return scores


    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def fit(self, train_loader, validation_loader=None):
        self.cuda() if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')
        
        for epoch in range(1, self.epochs + 1):  
            self.train()          
            total_loss = []
            for i, (seq, target, lens) in enumerate(train_loader):
                self.optimizer.zero_grad()
                scores = self.forward(seq.to(self.device), lens)
                loss = self.loss_function(torch.log(scores.clamp(min=1e-9)), target.squeeze().to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())

            s = ''
            if validation_loader:
                valid_loss = self.evaluate(validation_loader)
                s = f'\tValidation Loss: {valid_loss:.4f}'
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

    def evaluate(self, validation_loader):
        self.eval()
        valid_loss = []
        for _, (seq, target_item, lens) in enumerate(validation_loader):
            scores = self.forward(seq.to(self.device), lens)
            tmp_loss = self.loss_function(torch.log(scores.clamp(min=1e-9)), target_item.squeeze().to(self.device))
            valid_loss.append(tmp_loss.item())
            # TODO other metrics

        return np.mean(valid_loss)
        

