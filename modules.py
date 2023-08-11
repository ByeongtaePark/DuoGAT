import torch
import torch.nn as nn
from utils import *
torch.manual_seed(1)
torch.cuda.manual_seed(1)

class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_features, batch_size, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.batch_size = batch_size

        self.embed_dim *= 2
        lin_input_dim = 2 * n_features
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)

        self.embed_lin2 = nn.Linear(n_features, round(n_features/2))
        self.lin2 = nn.Linear(2*round(n_features/2), 2*round(n_features/2))

        self.embed_lin3 = nn.Linear(n_features, round(n_features/4))
        self.lin3 = nn.Linear(2*round(n_features/4), 2*round(n_features/4))

        self.attention_lin = nn.Linear(6, 1)
        self.mulatt_lin = nn.Linear(3, 1)
        
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        self.a2 = nn.Parameter(torch.empty((2 * round(n_features/2), 1)))
        self.a3 = nn.Parameter(torch.empty((2 * round(n_features/4), 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.window_size, self.window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dif_x):
        x = torch.tensor(x, dtype = torch.float32)
        dif_x = torch.tensor(dif_x, dtype = torch.float32)

        # time-series graph
        a_input = self._make_attention_input(x,1)  
        a_input = self.leakyrelu(self.lin(a_input))   
        e = torch.matmul(a_input, self.a).squeeze(3)  

        x2 = self.embed_lin2(x)
        a2_input = self._make_attention_input(x2,2)  
        a2_input = self.leakyrelu(self.lin2(a2_input))  
        e2 = torch.matmul(a2_input, self.a2).squeeze(3)  

        x3 = self.embed_lin3(x)
        a3_input = self._make_attention_input(x3,4)  
        a3_input = self.leakyrelu(self.lin3(a3_input))  
        e3 = torch.matmul(a3_input, self.a3).squeeze(3)  

        attention = torch.softmax(e, dim=2)
        attention2 = torch.softmax(e2, dim=2)
        attention3 = torch.softmax(e3, dim=2)

        bat_size = attention.shape[0]

        ts_attention = torch.dropout(attention, self.dropout, train=self.training)
        ts_attention2 = torch.dropout(attention2, self.dropout, train=self.training) 
        ts_attention3 = torch.dropout(attention3, self.dropout, train=self.training)

        # differecned graph
        a_input = self._make_attention_input(dif_x,1) 
        a_input = self.leakyrelu(self.lin(a_input)) 
        e = torch.matmul(a_input, self.a).squeeze(3)

        dif_x2 = self.embed_lin2(dif_x)
        a2_input = self._make_attention_input(dif_x2,2) 
        a2_input = self.leakyrelu(self.lin2(a2_input))
        e2 = torch.matmul(a2_input, self.a2).squeeze(3)

        dif_x3 = self.embed_lin3(dif_x)
        a3_input = self._make_attention_input(dif_x3,4)
        a3_input = self.leakyrelu(self.lin3(a3_input))
        e3 = torch.matmul(a3_input, self.a3).squeeze(3)

        attention = torch.softmax(e, dim=2)
        attention2 = torch.softmax(e2, dim=2)
        attention3 = torch.softmax(e3, dim=2)

        bat_size = attention.shape[0]

        dif_attention = torch.dropout(attention, self.dropout, train=self.training)
        dif_attention2 = torch.dropout(attention2, self.dropout, train=self.training)
        dif_attention3 = torch.dropout(attention3, self.dropout, train=self.training)


        ts_attention=torch.unsqueeze(ts_attention,3)
        ts_attention2=torch.unsqueeze(ts_attention2,3)
        ts_attention3=torch.unsqueeze(ts_attention3,3)
        dif_attention=torch.unsqueeze(dif_attention,3)
        dif_attention2=torch.unsqueeze(dif_attention2,3)
        dif_attention3=torch.unsqueeze(dif_attention3,3)

        attention_ts = torch.cat((ts_attention,ts_attention2,ts_attention3),3)
        attention_dif = torch.cat((dif_attention,dif_attention2,dif_attention3),3)
        
        attention_ts_mul = self.leakyrelu(self.mulatt_lin(attention_ts))
        attention_dif_mul = self.leakyrelu(self.mulatt_lin(attention_dif))

        attention_ts_mul=torch.squeeze(attention_ts_mul)
        attention_dif_mul=torch.squeeze(attention_dif_mul)
            
        if self.batch_size == bat_size:
            weight_edge = get_weight_matrix(bat_size,self.num_nodes)
            weight_matrix = torch.Tensor(weight_edge)
            weight_matrix = weight_matrix.to('cuda')
            attention_ts_mul = torch.mul(attention_ts_mul, weight_matrix)
                        
        else:
            weight_edge2 = get_weight_matrix(bat_size,self.num_nodes)
            weight_matrix2 = torch.Tensor(weight_edge2)
            weight_matrix2 = weight_matrix2.to('cuda')
            attention_ts_mul = torch.mul(attention_ts_mul, weight_matrix2)

        h_ts = self.sigmoid(torch.matmul(attention_ts_mul, x))
        h_dif = self.sigmoid(torch.matmul(attention_dif_mul, x))

        return h_ts, h_dif

    def _make_attention_input(self, v, div_embed):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        return combined.view(v.size(0), K, K, 2 * round(self.n_features/div_embed))

class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]
        return out, h

class Forecasting_Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()

        layers = [nn.Linear(in_dim, hid_dim)]
        for i in range(n_layers-1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
            
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
            
        return self.layers[-1](x)