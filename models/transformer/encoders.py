from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.transformer.gaussian_kernel import get_gaussian_filter

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=512, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):            
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=512, dropout=.1,
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

    def forward(self, input, attention_weights=None):
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))


        outs = torch.cat(outs, 1)
        return outs, attention_mask



class MemoryAugmentedEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_in=512, d_model=512, dropout=.1, **kwargs):  
        super(MemoryAugmentedEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.MLencoder = MultiLevelEncoder(N, padding_idx)

    def forward(self, input, attention_weights=None): 
        out = F.relu(self.fc(input)) 
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = self.MLencoder(out, attention_weights=attention_weights)
        return out


'''
Paper : Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
'''
class MemoryAugmentedEncoder_CBS(nn.Module):
    def __init__(self, N, padding_idx, d_in=512, d_model=512, dropout=.1, std=1, **kwargs):  
        super(MemoryAugmentedEncoder_CBS, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.std = std

        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.MLencoder = MultiLevelEncoder(N, padding_idx)
        
    def get_new_kernels(self, epoch_count, kernel_sizex, kernel_sizey, decay_epoch, std_factor):
        if epoch_count % decay_epoch == 0 and epoch_count is not 0:
            self.std *= std_factor
        self.kernel1 = get_gaussian_filter(
                kernel_sizex=kernel_sizex,
                kernel_sizey=kernel_sizey,
                sigma=self.std,
                channels=6,
            )
    def forward(self, input, attention_weights=None):
        out = self.fc(input)
        out = F.relu(self.layer_norm(self.kernel1(out))) 
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = self.MLencoder(out, attention_weights=attention_weights)
        return out











