import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

#class Residual(Module):
#    def __init__(self, hidden_size):
#        super().__init__()
#        self.hidden_size = hidden_size
#        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#        self.dp = nn.Dropout(p=0.2)
#        self.drop = True
#
#    def forward(self, x):
#        residual = x  # keep original input
#        x = F.relu(self.d1(x))
#        if self.drop:
#            x = self.d2(self.dp(x))
#        else:
#            x = self.d2(x)
#        out = residual + x
#        return out
        
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        #hy = newgate + inputgate * (hidden - newgate)
        hy = hidden - inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCSAN(Module):
    def __init__(self, n_node, conf, logger):
        '''
        SR-GNN model class: https://ojs.aaai.org/index.php/AAAI/article/view/3804/3682

        Parameters
        ----------
        n_node : int
            the number of items
        step : int
            gnn propogation steps
        item_embedding_dim : int
            the dimension of hidden layers, also known as item embedding
        lr : float
            learning rate
        l2 : float
            L2-regularization term
        lr_dc_step : int
            Period of learning rate decay
        lr_dc : float
            Multiplicative factor of learning rate decay, by default 1 0.1
        '''    
        super(GCSAN, self).__init__()
        self.hidden_size = conf['item_embedding_dim']  # this hidden size is item embedding in SRGNN
        self.n_node = n_node + 1
        self.batch_size = conf['batch_size']
        self.weight = conf['weight'] # hyper-parameter [0.4, 0.8]
        self.blocks = conf['blocks'] # hyper-parameter [1,2,3,4]
        self.dropout_rate = conf['dropout_hidden']

        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=conf['step'])
        # gcsan addition to srgnn
#        self.rn = Residual(self.hidden_size)
#        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        for _ in range(self.blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_size,
                                                            1,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=conf['learning_rate'], weight_decay=conf['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=conf['lr_dc_step'], gamma=conf['lr_dc'])
        self.reset_parameters()

        self.logger = logger
        self.epochs = conf['epochs']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, residual=True):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        #mask_self = mask.repeat(1, mask.shape[1]).view(-1, mask.shape[1], mask.shape[1])

#        for k in range(self.blocks):
#            # single attention -> point-wise feed forward -> residual connection
#            attn_output = attn_output.transpose(0,1)
#            attn_output, attn_output_weights = self.multihead_attn(attn_output, attn_output, attn_output)
#            attn_output = attn_output.transpose(0,1)
#            # 加上 residual network
#            if residual:
#                attn_output = self.rn(attn_output)
                
        attn_output = hidden # batch_size * seq_len * dim
        timeline_mask = torch.BoolTensor(mask.cpu()==0).to(self.device)
        attn_output *=  ~timeline_mask.unsqueeze(-1)
        for i in range(self.blocks):
            # Single Attention
            attn_output = attn_output.transpose(0,1)
            Q = self.attention_layernorms[i](attn_output)
            mha_outputs, _ = self.attention_layers[i](Q, attn_output, attn_output, key_padding_mask=timeline_mask)
            mha_outputs = torch.transpose(mha_outputs, 0, 1)
            attn_output = mha_outputs
            
            # Point-Wise FeedForward
            mha_outputs = self.forward_layernorms[i](mha_outputs)
            mha_outputs = self.forward_layers[i](mha_outputs)
            
            # Residual
            attn_output += mha_outputs
            attn_output *=  ~timeline_mask.unsqueeze(-1)
            
        attn_output = self.last_layernorm(attn_output)
        
        hn = attn_output[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # use last one as global interest

        a = self.weight *hn + (1-self.weight)*ht  # hyper-parameter w
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

    def _forward(self, i, data):
        alias_inputs, A, items, mask, targets = data.get_slice(i)
        alias_inputs = torch.Tensor(alias_inputs).long().to(self.device)
        items = torch.Tensor(items).long().to(self.device)
        A = torch.Tensor(A).float().to(self.device)
        mask = torch.Tensor(mask).long().to(self.device)
        hidden = self.forward(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return targets, self.compute_scores(seq_hidden, mask)

    def fit(self, train_data, validation_data=None):
        self.cuda() if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []
            slices = train_data.generate_batch(self.batch_size)
            for i, j in zip(slices, np.arange(len(slices))):
                self.optimizer.zero_grad()

                targets, scores = self._forward(i, train_data)

                targets = torch.Tensor(targets).long().to(self.device)
                loss = self.loss_function(scores, (targets - 1).squeeze())
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            s = ''
            if validation_data:
                valid_loss = self.evaluate(validation_data)
                s = f'\tValidation Loss: {valid_loss:.4f}'
            self.logger.info(f'Training Epoch: {epoch}\tLoss: {np.mean(total_loss):.4f}' + s)

    def predict(self, test_data, k=15):
        self.logger.info('Start predicting...')
        self.eval()
        slices = test_data.generate_batch(self.batch_size)
        preds, last_item = torch.tensor([]), torch.tensor([])
        for i in slices:
            targets, scores = self._forward(i, test_data)
            sub_scores = scores.topk(k)[1]
            sub_scores = sub_scores + 1 # +1 change to actual code we did before

            preds = torch.cat((preds, sub_scores.cpu()), 0)
            last_item = torch.cat((last_item, torch.tensor(targets)), 0)

        return preds, last_item

    def evaluate(self, validation_data):
        self.eval()
        valid_loss = []
        slices = validation_data.generate_batch(self.batch_size)
        for i, _ in zip(slices, np.arange(len(slices))):
            targets, scores = self._forward(i, validation_data)
            targets = torch.Tensor(targets).long().to(self.device)
            loss = self.loss_function(scores, (targets - 1).squeeze())
            valid_loss.append(loss.item())

        return np.mean(valid_loss)
