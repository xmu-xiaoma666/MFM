from torch.nn import functional as F
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention
from ..relative_embedding import GridRelationalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None, pos=None):
        att = self.mhatt(queries, keys, values, relative_pos, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)

        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        relative_geometry_embeddings = GridRelationalEmbedding(input.shape[0])
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [w(flatten_relative_geometry_embeddings).view(box_size_per_head) for w in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        grid2grid = F.relu(relative_geometry_weights)
        out = input
        for l in self.layers:
            out = l(out, out, out, grid2grid, attention_mask, attention_weights, pos=pos)

        return out, attention_mask


from numpy import math
import numpy as np
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        rowPE = torch.zeros(max_len,max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        rowPE[ :,:,0::2] = torch.sin(position * div_term)
        rowPE[ :,:, 1::2] = torch.cos(position * div_term)
        colPE=rowPE.transpose(1, 0)
        rowPE = rowPE.unsqueeze(0)
        colPE = colPE.unsqueeze(0)
        self.rowPE=rowPE.cuda()
        self.colPE=colPE.cuda()

    def forward(self, x):
        feat=x
        bs,gs,dim=feat.shape
        feat=feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
        feat = feat + self.rowPE[:, :int(np.sqrt(gs)), :int(np.sqrt(gs)),  :dim ]+ self.colPE[:,  :int(np.sqrt(gs)),  :int(np.sqrt(gs)),  :dim ]
        feat=feat.view(bs,-1,dim)
        return self.dropout(feat)


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pe=PositionalEncoding(d_model=d_in,dropout=0)

    def forward(self, input, attention_weights=None):
        feat=self.pe(input)
        mask = (torch.sum(feat, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(feat))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
