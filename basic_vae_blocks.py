import torch
import torch.nn as nn
import torch.nn.functional as F
from music_base import VALID_MIDI_COUNT

WHOLE_CONDITION_DIM=12+12+12+4+4
WHOLE_X_DIM=2+VALID_MIDI_COUNT # silence, sustain, MIDIs

def init_hidden(batch_size,hidden_dim,z,layer_num=1,direction=2):
    h_0=torch.zeros(direction*layer_num,batch_size,hidden_dim//2,device=z.device)
    c_0=torch.zeros(direction*layer_num,batch_size,hidden_dim//2,device=z.device)
    return h_0,c_0

def hard_max(x):
    '''
    :param x: (batch_size,feature_dim)
    :return: (batch_size,feature_dim), one-hot version
    '''
    idx=x.max(1)[1]
    range_obj=torch.arange(x.shape[0],dtype=torch.long,device=x.device)
    result=torch.zeros_like(x,device=x.device)
    result[range_obj,idx]=1.0
    return result

class LocalEncoder(nn.Module):
    # todo: fix me!
    def __init__(self,half_hidden_size,num_layers):
        super(LocalEncoder, self).__init__()
        self.hidden_dim=half_hidden_size*2
        self.num_layers=num_layers
        self.lstm=nn.LSTM(
            input_size=(WHOLE_X_DIM+WHOLE_CONDITION_DIM),
            hidden_size=self.hidden_dim//2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        '''

        :param x: (batch_size,length_in_notes,x_dim+condition_dim)
        :return: (batch_size,length_in_bars,hidden_dim*num_layers)
        '''
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=x.view((batch_size*seq_length//16,16,x.shape[2]))
        _,(h,c)=self.lstm(x,init_hidden(batch_size*seq_length//16,self.hidden_dim,x,self.num_layers))
        h=h.transpose(0,1).contiguous().view((batch_size,seq_length//16,self.hidden_dim*self.num_layers))
        return h

class LocalDecoder(nn.Module):

    def __init__(self,hidden_size,num_layers):
        super(LocalDecoder, self).__init__()
        self.hidden_dim=hidden_size

        self.num_layers=num_layers
        self.lstms=nn.Sequential(
            *[nn.LSTMCell(input_size=WHOLE_X_DIM+WHOLE_CONDITION_DIM if i==0 else self.hidden_dim,
                          hidden_size=self.hidden_dim)
              for i in range(self.num_layers)])

        self.final_fc=nn.Linear(self.hidden_dim,WHOLE_X_DIM)
        self.teacher_forcing=1.0

    def forward(self,z,x):
        '''
        :param z: (batch_size,length_in_bars,hidden_dim)
        :param x: (batch_size,length_in_notes,x_dim+condition_dim)
        :return: (batch_size,length_in_notes,x_dim)
        '''
        n_step=16
        batch_size=z.shape[0]
        seq_length=z.shape[1]*16
        x=x.view((batch_size*seq_length//16,16,x.shape[2]))
        z=z.view((batch_size*seq_length//16,z.shape[2]))
        y=x[:,:,-WHOLE_CONDITION_DIM:]
        h=[z for _ in range(self.num_layers)]
        c=[torch.zeros((batch_size*seq_length//16,self.hidden_dim),dtype=z.dtype,device=z.device) for _ in range(self.num_layers)]
        o=torch.zeros((batch_size*seq_length//16,WHOLE_X_DIM),dtype=z.dtype,device=z.device)
        o[:,1]=1.0 # todo: sustain or silence?
        result=[]
        for i in range(n_step):
            # todo: weather concat z here
            for k in range(self.num_layers):
                if(k==0):
                    o=torch.cat((o,y[:,i,:]),dim=1)
                h[k],c[k]=self.lstms[k](o,(h[k],c[k]))
                o=h[k]
            o=F.log_softmax(self.final_fc(o),1)
            result.append(o)
            if(self.training and torch.rand(1).item()<self.teacher_forcing):
                o=x[:,i,:WHOLE_X_DIM]
            else:
                o=hard_max(o)
        return torch.stack(result,1).view((batch_size,seq_length,WHOLE_X_DIM))

class Reparameterizer(nn.Module):

    def __init__(self,input_hidden_size,z_dim):
        super(Reparameterizer, self).__init__()
        self.z_dim=z_dim
        self.linear_mu=nn.Linear(input_hidden_size,z_dim)
        self.linear_sigma=nn.Linear(input_hidden_size,z_dim)
        self.supress_warning=False

    def forward(self, z, is_training=None):
        '''
        :param z: (..., input_hidden_size)
        :return: (..., z_dim)
        '''
        if(is_training is None):
            if(not self.supress_warning):
                print('[Warning] The reparameterizer now requires a new explicit parameter is_training. Please fix your code.')
                self.supress_warning=True
            is_training=self.training
        mu=self.linear_mu(z)
        if(is_training):
            logvar=self.linear_sigma(z)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu),mu,logvar
        else:
            return mu,None,None

    def collect_statistics(self, z):
        mu=self.linear_mu(z)
        logvar=self.linear_sigma(z)
        return mu,logvar


def interp_path(z1,z2,interpolation_count):
    import numpy as np
    result_shape=z1.shape
    z1=z1.reshape(-1)
    z2=z2.reshape(-1)
    def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1
    def slerp2(p0,p1,t):
        omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega)[:,None] / so * p0[None] + np.sin(t*omega)[:,None]/so * p1[None]
    percentages=np.linspace(0.0,1.0,interpolation_count)
    normalized_z1=z1/np.linalg.norm(z1)
    normalized_z2=z2/np.linalg.norm(z2)
    #dirs=np.stack([slerp(normalized_z1,normalized_z2,t) for t in percentages])
    dirs=slerp2(normalized_z1,normalized_z2,percentages)
    length=np.linspace(np.log(np.linalg.norm(z1)),np.log(np.linalg.norm(z2)),interpolation_count)
    return (dirs*np.exp(length[:,None])).reshape([interpolation_count]+list(result_shape))
