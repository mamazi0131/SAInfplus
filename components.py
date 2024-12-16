import torch
import torch.nn as nn
import torch.nn.functional as F
from basemodel import BaseModel
from torch_geometric.nn import GATConv
from math import sqrt
from torch_geometric.nn import HypergraphConv

# vanilla transformer
class TransformerModel(BaseModel):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output

class GraphEncoder(BaseModel):
    def __init__(self, input_size, output_size, drop_p):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size, dropout=drop_p)
        self.layer2 = GATConv(output_size, input_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x

class DiagonalAttention(BaseModel):
    def __init__(self, dim_in, dim_k, dim_v):
        super(DiagonalAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, pair_rep, candidate_region_rep):
        # using pair_query update canidate_region_rep
        # take the diagonal of the attention matrix
        q = self.linear_q(pair_rep)   # batch, n, dim_k
        k = self.linear_k(candidate_region_rep)  # batch, n, dim_k
        v = self.linear_v(candidate_region_rep)  # batch, n, dim_v
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        ones = torch.eye(dist.shape[-1]).unsqueeze(0).expand_as(dist).cuda()
        dist = dist*ones
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att

class Attention(BaseModel):
    def __init__(self, dim_in, dim_k, dim_v):
        super(Attention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, pair_rep, candidate_region_rep):
        # update query
        q = self.linear_q(candidate_region_rep)   # batch, n, dim_k
        k = self.linear_k(pair_rep)  # batch, n, dim_k
        v = self.linear_v(pair_rep)  # batch, n, dim_v
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att

class PointerNet(BaseModel):
    def __init__(self, dim_in, dim_k, dim_v):
        super(PointerNet, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, pair_rep, candidate_region_rep):
        q = self.linear_q(pair_rep)   # batch, n, dim_k
        k = self.linear_k(candidate_region_rep)  # batch, n, dim_k
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        return dist
   
class FeedForwardNetwork(BaseModel):
    '''
    Using nn.Conv1d replace nn.Linear to implements FFN.
    '''
    def __init__(self, d_model, d_ff, p_drop):
        super(FeedForwardNetwork, self).__init__()
        # self.ff1 = nn.Linear(d_model, d_ff)
        # self.ff2 = nn.Linear(d_ff, d_model)
        self.ff1 = nn.Conv1d(d_model, d_ff, 1)
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=p_drop)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = x.transpose(1, 2) # [batch, d_model, seq_len]
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = x.transpose(1, 2) # [batch, seq_len, d_model]

        return self.layer_norm(residual + x)
    
class MultiHeadAttention(BaseModel):
    def __init__(self,d_model, d_k, d_v, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        # do not use more instance to implement multihead attention
        # it can be complete in one matrix
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # we can't use bias because there is no bias term in formular
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        To make sure multihead attention can be used both in encoder and decoder, 
        we use Q, K, V respectively.
        input_Q: [batch, len_q, d_model]
        input_K: [batch, len_k, d_model]
        input_V: [batch, len_v, d_model]
        '''
        residual, batch = input_Q, input_Q.size(0)

        # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view --> 
        # [batch, len_q, n_heads, d_k,] -- transpose --> [batch, n_heads, len_q, d_k]

        Q = self.W_Q(input_Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]

        # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
        prob, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
        prob = prob.view(batch, -1, self.n_heads * self.d_v).contiguous() # [batch, len_q, n_heads * d_v]

        output = self.fc(prob) # [batch, len_q, d_model]

        return self.layer_norm(residual + output), attn

class EncoderLayer(BaseModel):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, encoder_input, encoder_pad_mask):
        '''
        encoder_input: [batch, source_len, d_model]
        encoder_pad_mask: [batch, n_heads, source_len, source_len]

        encoder_output: [batch, source_len, d_model]
        attn: [batch, n_heads, source_len, source_len]
        '''
        encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
        encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

        return encoder_output, attn

class Expert(BaseModel):
    def __init__(self,input_dim, output_dim, dropout=0.1):
        super(Expert, self).__init__()
        self.expert_layer = nn.Sequential(
                            nn.Linear(input_dim, output_dim),
                            nn.ReLU()
                            )  

    def forward(self, x):
        out = self.expert_layer(x)
        return out

class MMOE(nn.Module):
    def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): 
        super(MMOE, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        # expert
        for i in range(n_expert):
            setattr(self, "expert_layer"+str(i+1), Expert(feature_dim,expert_dim)) 
        self.expert_layers = [getattr(self,"expert_layer"+str(i+1)) for i in range(n_expert)]
        
        # gate
        for i in range(n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(feature_dim, n_expert),
                                        					   nn.Softmax(dim=1))) 
        self.gate_layers = [getattr(self,"gate_layer"+str(i+1)) for i in range(n_task)]
        
    def forward(self, x):
        if self.use_gate:
            output_expert = [expert(x) for expert in self.expert_layers]
            output_expert = torch.cat(([e.unsqueeze(1) for e in output_expert]),dim = 1) # (bs,n_expert,expert_dim)

            output_gate = [gate(x) for gate in self.gate_layers]     # n_task*(bs,n_expert)

            towers = []
            for i in range(self.n_task):
                gate = output_gate[i].unsqueeze(-1)  # (bs,n_expert,1)
                tower = torch.matmul(output_expert.transpose(1,2),gate)  # (bs,expert_dim,1)
                towers.append(tower.transpose(1,2).squeeze(1))           # (bs,expert_dim)
        else:
            output_expert = [expert(x) for expert in self.expert_layers]
            towers = sum(output_expert)/len(output_expert)
        return towers


  




    