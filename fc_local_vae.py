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

TRAIN_BAR_COUNT=1

class FCEncoder(NetworkBehavior):

    def __init__(self,output_dim):
        super(FCEncoder, self).__init__()
        self.fc1=nn.Linear(WHOLE_X_DIM*TRAIN_BAR_COUNT*16,512)
        self.fc2=nn.Linear(512,384)
        self.fc3=nn.Linear(384,output_dim)


    def forward(self, x):
        '''

        :param x: (batch_size,length_in_notes,x_dim+condition_dim)
        :return: (batch_size,length_in_bars,hidden_dim*num_layers)
        '''
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=x[:,:,:WHOLE_X_DIM].contiguous().view((batch_size,seq_length//16,16*WHOLE_X_DIM))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class FCDecoder(NetworkBehavior):

    def __init__(self,input_dim):
        super(FCDecoder, self).__init__()
        self.fc1=nn.Linear(input_dim,384)
        self.fc2=nn.Linear(384,512)
        self.fc3=nn.Linear(512,WHOLE_X_DIM*TRAIN_BAR_COUNT*16)

    def forward(self, z, x):
        n_step=16
        batch_size=z.shape[0]
        seq_length=z.shape[1]*16
        z=z.view((batch_size,seq_length//16,z.shape[2]))
        z=F.relu(self.fc1(z))
        z=F.relu(self.fc2(z))
        z=self.fc3(z)
        return z.view((batch_size,seq_length,WHOLE_X_DIM))


class FCLocalVAE(NetworkBehavior):

    def __init__(self,z_dim,kl_coef=1.0):
        super(FCLocalVAE, self).__init__()
        self.z_dim=z_dim
        self.kl_coef=kl_coef
        self.encoder=FCEncoder(self.z_dim)
        self.reparameterizer=Reparameterizer(self.z_dim,self.z_dim)
        self.decoder=FCDecoder(self.z_dim)

    def forward(self,x):
        z=self.encoder(x)
        z,mu,logvar=self.reparameterizer(z)
        #z=self.decoder_init_linear(z)
        x_recon=self.decoder(z,x)
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
        z=self.encoder(data)
        z,mu,logvar=self.reparameterizer(z)
        assert(logvar is None)
        return z.cpu().detach().numpy()

    def inference_decode(self,z,input):
        if(len(input.shape)==2):
            input=input.view((1,input.shape[0],input.shape[1]))
        recon_x=self.decoder(z,input)
        result=F.softmax(recon_x.view((-1,WHOLE_X_DIM)),dim=1).cpu().detach().numpy()
        return result


def get_pretrained_dict(model_name='fc_vae_v0.2_256_s0.cp'):
    from mir.nn.train import NetworkInterface
    model=FCLocalVAE(256,0.1)
    net=NetworkInterface(model,model_name,load_checkpoint=False)
    return net.net.encoder.state_dict(),net.net.decoder.state_dict()


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
    storage_x=FramedRAMDataStorage('D:/hooktheory_gen_update_4')
    storage_x.load_meta()
    song_count=storage_x.get_length()
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
    train_provider=FramedDataProvider(train_sample_length=TRAIN_BAR_COUNT,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,
                                      num_workers=1,allow_truncate=True,average_samples_per_song=1)
    train_provider.link(storage_x,GenPitchShifter(fix_bar_count=TRAIN_BAR_COUNT),subrange=train_indices)

    val_provider=FramedDataProvider(train_sample_length=TRAIN_BAR_COUNT,shift_low=SHIFT_LOW,shift_high=SHIFT_HIGH,
                                    num_workers=1,allow_truncate=True,average_samples_per_song=1)

    val_provider.link(storage_x,GenPitchShifter(fix_bar_count=TRAIN_BAR_COUNT),subrange=val_indices)
    trainer=NetworkInterface(FCLocalVAE(256,0.1),'fc_vae_v0.2_256_s%d'%slice_id,load_checkpoint=True)
    trainer.train_supervised(train_provider,val_provider,batch_size=64,
                                     learning_rates_dict={1e-3:12,1e-4:6},round_per_print=10,round_per_val=50,round_per_save=200)

