from modules import Transformer,TransformerEncoderLayer,TransformerDecoderLayer
import torch.nn as nn
import torch.nn.functional as F
from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage
from mir.nn.data_provider import FramedDataProvider
from mir.nn.train import NetworkBehavior,NetworkInterface
import torch
import numpy as np
from music_base import SHIFT_LOW,SHIFT_HIGH,MIDI_BASE,VALID_MIDI_COUNT
from basic_vae_blocks import LocalEncoder,LocalDecoder,WHOLE_X_DIM,WHOLE_CONDITION_DIM,Reparameterizer
from pitch_shifter import GenPitchShifter
from fc_local_vae import FCDecoder,FCEncoder,get_pretrained_dict
from basic_vae_blocks import hard_max

TRAIN_BAR_COUNT=16
TRANSFORMER_LAYER=3
N_HEADS=4

def create_mask(batch_size,seq_length,device,type):
    def create_attention_mask_tril(batch_size,seq_len,device):
        x = torch.ones(
            seq_len, seq_len, device=device).tril(-1).transpose(0, 1)
        return x.repeat(batch_size, 1, 1).byte()

    def create_attention_mask_diagonal(batch_size,seq_len,device):
        x = 1.-torch.eye(
            seq_len, seq_len, device=device)
        return x.repeat(batch_size, 1, 1).byte()
    if(type==0):
        return None
    elif(type==1):
        return create_attention_mask_tril(batch_size,seq_length,device)
    elif(type==2):
        return create_attention_mask_diagonal(batch_size,seq_length,device)
    else:
        raise NotImplementedError()


class TransformerHierarchicalEncoder(nn.Module):
    def __init__(self,local_dim,train_bar_count,mask_type=0):
        super(TransformerHierarchicalEncoder, self).__init__()
        self.local_encoder=FCEncoder(local_dim)
        self.pe=nn.Embedding(train_bar_count,local_dim)
        self.transformer_encoder=nn.Sequential(*[TransformerEncoderLayer(
            dim_m=local_dim,dim_q_k=local_dim,dim_v=local_dim,
            n_heads=N_HEADS,dim_i=local_dim,dropout=0.0) for _ in range(TRANSFORMER_LAYER)])
        self.mask_type=mask_type
        self.train_bar_count=train_bar_count

    def forward(self, x, return_attn_weight=False):
        '''
        :param x: (batch_size,length_in_notes,x_dim+condition_dim)
        :return: (batch_size,global_hidden_dim*num_global_layers)
        '''
        batch_size=x.shape[0]
        h=self.local_encoder(x)
        positions=torch.arange(self.train_bar_count,dtype=torch.long,device=h.device).view((1,-1))
        pe=self.pe(positions)
        attn_weights=[]
        if(TRANSFORMER_LAYER>0):
            z=h+pe
            mask=create_mask(batch_size,self.train_bar_count,z.device,self.mask_type)
            for transformer_encoder in self.transformer_encoder:
                if(return_attn_weight):
                    z,attn_weight=transformer_encoder(z,mask=mask,return_attn_weight=True)
                    attn_weights.append(attn_weight)
                else:
                    z=transformer_encoder(z,mask=mask,return_attn_weight=False)
        else:
            z=h
        if(return_attn_weight):
            return z.view((batch_size,-1)),h.view((batch_size,-1)),attn_weights
        else:
            return z.view((batch_size,-1)),h.view((batch_size,-1))

class TransformerHierarchicalDecoder(nn.Module):

    def __init__(self,local_dim,train_bar_count,mask_type_inter=0,mask_type_inner=1):
        super(TransformerHierarchicalDecoder, self).__init__()
        self.local_dim=local_dim
        self.pe=nn.Embedding(train_bar_count,local_dim)
        self.transformer_decoders=nn.Sequential(*[TransformerDecoderLayer(
            dim_m=local_dim,dim_q_k=local_dim,dim_v=local_dim,
            n_heads=N_HEADS,dim_i=local_dim,dropout=0.0) for _ in range(TRANSFORMER_LAYER)])
        self.local_decoder=FCDecoder(local_dim)
        self.mask_type_inter=mask_type_inter
        self.mask_type_inner=mask_type_inner
        self.train_bar_count=train_bar_count

    def forward(self,z,x,h):
        '''
        :param z: (batch_size,input_hidden_dim)
        :param x: (batch_size,length_in_notes,x_dim+condition_dim)
        :return: (batch_size,length_in_notes,x_dim)
        '''
        batch_size=z.shape[0]
        seq_length=x.shape[1]
        z=z.view((batch_size,self.train_bar_count,self.local_dim))
        h=h.view((batch_size,self.train_bar_count,self.local_dim))
        padded_h=torch.zeros_like(h,device=h.device,dtype=h.dtype)
        padded_h[:,1:,:]=h[:,:-1,:]
        #z[:,(TRAIN_BAR_COUNT//2):,:]=0 # dropout
        positions=torch.arange(self.train_bar_count,dtype=torch.long,device=z.device).view((1,-1))
        d=self.pe(positions).expand((batch_size,-1,-1))+padded_h
        mask=create_mask(batch_size,self.train_bar_count,d.device,self.mask_type_inner)
        extra_mask=create_mask(batch_size,self.train_bar_count,d.device,self.mask_type_inter)
        if(TRANSFORMER_LAYER>0):
            for transformer_decoder in self.transformer_decoders:
                d=transformer_decoder(d,z,mask,extra_mask=extra_mask)
        else:
            d=z
        return self.local_decoder(d,x)
