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
from transformer_vae import TransformerHierarchicalEncoder,TransformerHierarchicalDecoder


TRAIN_BAR_COUNT=8
TRANSFORMER_LAYER=3
N_HEADS=4

class TransformerSequentialVAE(NetworkBehavior):

    def __init__(self,local_dim,z_dim,kl_coef=1.0,mask_type_enc=1,mask_type_inter=2,mask_type_dec=1,pretrain=False):
        super(TransformerSequentialVAE, self).__init__()
        self.z_dim=z_dim
        self.kl_coef=kl_coef
        self.local_dim=local_dim
        self.encoder=TransformerHierarchicalEncoder(local_dim,TRAIN_BAR_COUNT,mask_type=mask_type_enc)
        self.reparameterizers=nn.Sequential(*[Reparameterizer(local_dim,self.z_dim) for _ in range(TRAIN_BAR_COUNT)])
        self.decoder_init_linears=nn.Sequential(*[nn.Linear(self.z_dim,local_dim) for _ in range(TRAIN_BAR_COUNT)])
        self.decoder=TransformerHierarchicalDecoder(local_dim,TRAIN_BAR_COUNT,mask_type_inter=mask_type_inter,mask_type_inner=mask_type_dec)
        if(pretrain):
            encoder_dict,decoder_dict=get_pretrained_dict()
            self.encoder.local_encoder.load_state_dict(encoder_dict)
            self.decoder.local_decoder.load_state_dict(decoder_dict)
    def forward(self,x):
        batch_size=x.shape[0]
        z,h=self.encoder(x)
        z=z.view((batch_size,TRAIN_BAR_COUNT,self.local_dim))
        h=h.view((batch_size,TRAIN_BAR_COUNT,self.local_dim))
        rep=[self.reparameterizers[i](z[:,i,:], is_training=True) for i in range(TRAIN_BAR_COUNT)]
        z=torch.stack([self.decoder_init_linears[i](rep[i][0]) for i in range(TRAIN_BAR_COUNT)],dim=1).view((batch_size,-1))
        if(len(rep)>0 and rep[0][2] is not None):
            mu=torch.stack([t[1] for t in rep],dim=1)
            logvar=torch.stack([t[2] for t in rep],dim=1)
        else:
            mu,logvar=None,None
        x_recon=self.decoder(z,x,h)
        return x_recon,mu,logvar

    def loss_function(self,x_recon,x_tag,mu,logvar):
        kld=(-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp()))
        ce=F.cross_entropy(x_recon,x_tag)
        return kld*self.kl_coef+ce

    def loss(self,input):
        data=input[:,:,1:]
        tag=input[:,:,0].type(torch.cuda.LongTensor if self.use_gpu else torch.LongTensor)
        recon_x,mu,logvar=self.feed(data)
        return self.loss_function(recon_x.view((-1,WHOLE_X_DIM)),tag.view((-1)),mu,logvar)

    def inference_encode(self,input):
        if(len(input.shape)==2):
            input=input.view((1,input.shape[0],input.shape[1]))
        data=input[:,:,1:]
        z,_=self.encoder(data)
        batch_size=z.shape[0]
        z=z.view((batch_size,TRAIN_BAR_COUNT,self.local_dim))
        rep=[self.reparameterizers[i](z[:,i,:], is_training=False) for i in range(TRAIN_BAR_COUNT)]
        if(len(rep)>0):
            assert(rep[0][2] is None)
        z=torch.stack([rep[i][0] for i in range(TRAIN_BAR_COUNT)],dim=1)
        return z.cpu().detach().numpy()

    def inference_decode(self,z,input):
        if(len(input.shape)==2):
            input=input.view((1,input.shape[0],input.shape[1]))
        data=input[:,:,1:].clone()
        batch_size=data.shape[0]
        h=torch.zeros((data.shape[0],TRAIN_BAR_COUNT,self.local_dim),device=z.device,dtype=z.dtype)
        result=[]
        z=torch.stack([self.decoder_init_linears[i](z[:,i,:]) for i in range(TRAIN_BAR_COUNT)],dim=1).view((batch_size,-1))
        data=data.view((batch_size,TRAIN_BAR_COUNT,16,data.shape[2]))
        for i in range(TRAIN_BAR_COUNT):
            recon_x=self.decoder(z,data,h).view((batch_size,TRAIN_BAR_COUNT,16,WHOLE_X_DIM))[:,i,:,:]
            result.append(recon_x)
            out=hard_max(recon_x.view((-1,WHOLE_X_DIM))).view((batch_size,16,WHOLE_X_DIM))
            #out=out.roll(1,2); print('Warning: wrong decoder condition conducted')
            data[:,i,:,:WHOLE_X_DIM]=out
            #print('Warning: h not modified')
            h[:,i,:]=self.encoder.local_encoder(data[:,i,:,:])
        result=torch.cat(result,dim=1)
        result=F.softmax(result.view((-1,WHOLE_X_DIM)),dim=1).cpu().detach().numpy()
        return result

    def inference_decode_cheat(self,z,input):
        data=input[None,:,1:]
        tag=input[None,:,0].type(torch.cuda.LongTensor if self.use_gpu else torch.LongTensor)
        recon_x,mu,logvar=self.feed(data)
        result=F.softmax(recon_x.view((-1,WHOLE_X_DIM)),dim=1).cpu().detach().numpy()
        return result

    def inference_accuracy(self,input):
        data=input[None,:,1:]
        tag=input[None,:,0].type(torch.cuda.LongTensor if self.use_gpu else torch.LongTensor)
        recon_x,mu,logvar=self.feed(data)
        result=torch.max(recon_x.view((-1,WHOLE_X_DIM)),dim=1)[1]
        return torch.sum(result==tag).item(),len(input)

    def collect_statistics(self,input):
        if(len(input.shape)==2):
            input=input.view((1,input.shape[0],input.shape[1]))
        data=input[:,:,1:]
        z,_=self.encoder(data)
        batch_size=z.shape[0]
        z=z.view((batch_size,TRAIN_BAR_COUNT,self.local_dim))
        rep=[self.reparameterizers[i].collect_statistics(z[:,i,:]) for i in range(TRAIN_BAR_COUNT)]
        mu=torch.stack([rep[i][0] for i in range(TRAIN_BAR_COUNT)],dim=1)
        logvar=torch.stack([rep[i][1] for i in range(TRAIN_BAR_COUNT)],dim=1)
        return mu.cpu().detach().numpy(),logvar.cpu().detach().numpy()

if __name__ == '__main__':
    TOTAL_SLICE_COUNT=5
    import sys
    try:
        slice_id=int(sys.argv[1])
        if(slice_id>=TOTAL_SLICE_COUNT or slice_id<0):
            raise Exception('Invalid input')
        print('Train on slice %d'%slice_id)
    except:
        print('Train on all slices')
        slice_id=-1
    try:
        maskids=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
    except:
        maskids=(1,1,1)
    storage_x=FramedRAMDataStorage('E:/dataset/hooktheory_gen_update_4')
    storage_x.load_meta()
    song_count=storage_x.get_length()
    kld=float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    is_valid=np.ones(song_count,dtype=np.bool)
    for i in range(song_count):
        if(i%TOTAL_SLICE_COUNT==slice_id):
            is_valid[i]=False
    train_indices=np.arange(song_count)[is_valid]
    val_indices=np.arange(song_count)[np.bitwise_not(is_valid)]
    print('Using %d samples to train'%len(train_indices))
    print('Using %d samples to val'%len(val_indices))
    if(len(val_indices)==0): # no validation
        val_indices=np.arange(0,song_count,50) # pretend that we have validation
    train_provider=FramedDataProvider(train_sample_length=-1,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,
                                      num_workers=0,allow_truncate=True,average_samples_per_song=1)
    train_provider.link(storage_x,GenPitchShifter(fix_bar_count=TRAIN_BAR_COUNT,erase_chord=True),subrange=train_indices)

    val_provider=FramedDataProvider(train_sample_length=-1,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,
                                    num_workers=0,allow_truncate=True,average_samples_per_song=1)
    print('Warning: only train the beginning!')
    val_provider.link(storage_x,GenPitchShifter(fix_bar_count=TRAIN_BAR_COUNT,erase_chord=True),subrange=val_indices)
    trainer=NetworkInterface(TransformerSequentialVAE(256,64,kld,maskids[0],maskids[1],maskids[2],True),
                             'transformer_sequential_vae_no_chord_v2.1_m%d%d%d_3_layer_kl%f_s%d'%(maskids[0],maskids[1],maskids[2],kld,slice_id),load_checkpoint=True)
    trainer.train_supervised(train_provider,val_provider,batch_size=16,
                                     learning_rates_dict={1e-4:30,1e-5:20,1e-6:10},round_per_print=100,round_per_val=500,round_per_save=2000)
