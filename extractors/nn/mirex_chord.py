import torch.nn.functional as F
from torch import nn
from mir.nn.train import NetworkBehavior,NetworkInterface
from mir.nn.data_storage import FramedRAMDataStorage,FramedH5DataStorage
from mir.nn.data_decorator import CQTPitchShifter,AbstractPitchShifter,NoPitchShifter
from mir.nn.data_provider import FramedDataProvider
import torch
from complex_chord import Chord,ChordTypeLimit,shift_complex_chord_array_list,complex_chord_chop,enum_to_dict,\
    TriadTypes,SeventhTypes,NinthTypes,EleventhTypes,ThirteenthTypes,complex_chord_chop_list
from train_eval_test_split import get_train_set_ids,get_test_set_ids,get_val_set_ids
from extractors.nn.network_ensemble import NetworkEnsemble
import os
from mir.extractors import ExtractorBase
from mir import io
from io_new.chordlab_io import ChordLabIO
import librosa
import numpy as np

SHIFT_LOW=-5
SHIFT_HIGH=6
SHIFT_STEP=3
SPEC_DIM=252
LSTM_TRAIN_LENGTH=1000

chord_limit=ChordTypeLimit(
    triad_limit=2,
    seventh_limit=2,
    ninth_limit=0,
    eleventh_limit=0,
    thirteenth_limit=0
)



class CNNFeatureExtractor(nn.Module):

    def norm_layer(self,channels):
        return nn.InstanceNorm2d(channels)

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.cdim1=16
        self.cdim2=32
        self.cdim3=64
        self.cdim4=80

        self.conv1a=nn.Conv2d(1,self.cdim1,(3,3),padding=(1,1))
        self.norm1a=self.norm_layer(self.cdim1)
        self.conv1b=nn.Conv2d(self.cdim1,self.cdim1,(3,3),padding=(1,1))
        self.norm1b=self.norm_layer(self.cdim1)
        self.conv1c=nn.Conv2d(self.cdim1,self.cdim1,(3,3),padding=(1,1))
        self.norm1c=self.norm_layer(self.cdim1)
        self.pool1=nn.MaxPool2d((1,3))
        self.conv2a=nn.Conv2d(self.cdim1,self.cdim2,(3,3),padding=(1,1))
        self.norm2a=self.norm_layer(self.cdim2)
        self.conv2b=nn.Conv2d(self.cdim2,self.cdim2,(3,3),padding=(1,1))
        self.norm2b=self.norm_layer(self.cdim2)
        self.conv2c=nn.Conv2d(self.cdim2,self.cdim2,(3,3),padding=(1,1))
        self.norm2c=self.norm_layer(self.cdim2)
        self.pool2=nn.MaxPool2d((1,3))
        self.conv3a=nn.Conv2d(self.cdim2,self.cdim3,(3,3),padding=(1,1))
        self.norm3a=self.norm_layer(self.cdim3)
        self.conv3b=nn.Conv2d(self.cdim3,self.cdim3,(3,3),padding=(1,1))
        self.norm3b=self.norm_layer(self.cdim3)
        self.pool3=nn.MaxPool2d((1,4))
        self.conv4a=nn.Conv2d(self.cdim3,self.cdim4,(3,3),padding=(1,0))
        self.norm4a=self.norm_layer(self.cdim4)
        self.conv4b=nn.Conv2d(self.cdim4,self.cdim4,(3,3),padding=(1,0))
        self.norm4b=self.norm_layer(self.cdim4)
        self.output_size=3*self.cdim4

    def forward(self, x):
        assert(len(x.shape)==3)
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=x.view((batch_size,1,seq_length,SPEC_DIM))
        x=F.selu(self.norm1a(self.conv1a(x)))
        x=F.selu(self.norm1b(self.conv1b(x)))
        x=F.selu(self.norm1c(self.conv1c(x)))
        x=self.pool1(x)
        x=F.selu(self.norm2a(self.conv2a(x)))
        x=F.selu(self.norm2b(self.conv2b(x)))
        x=F.selu(self.norm2c(self.conv2c(x)))
        x=self.pool2(x)
        x=F.selu(self.norm3a(self.conv3a(x)))
        x=F.selu(self.norm3b(self.conv3b(x)))
        x=self.pool3(x)
        x=F.selu(self.norm4a(self.conv4a(x)))
        x=F.selu(self.norm4b(self.conv4b(x)))
        x=x.transpose(1,2).contiguous().view((batch_size,seq_length,self.output_size))
        return x

class ChordNet(NetworkBehavior):

    def __init__(self):
        super(ChordNet, self).__init__()
        self.audio_feature_block=CNNFeatureExtractor()

        self.condition_linear=nn.Linear(self.audio_feature_block.output_size+12+chord_limit.triad_limit+12,128)

        self.hidden_dim1=192
        self.lstm1=nn.LSTM(
            input_size=self.audio_feature_block.output_size,
            hidden_size=self.hidden_dim1//2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.output_dim1=chord_limit.triad_limit*12+2+12
        self.output_dim2=chord_limit.seventh_limit+chord_limit.ninth_limit+chord_limit.eleventh_limit+chord_limit.thirteenth_limit+4
        self.final_fc1=nn.Linear(self.hidden_dim1,self.output_dim1+self.output_dim2)

    def init_hidden(self,batch_size,hidden_dim):
        c_0=torch.zeros(2,batch_size,hidden_dim//2)
        h_0=torch.zeros(2,batch_size,hidden_dim//2)
        if(self.use_gpu):
            c_0=c_0.cuda()
            h_0=h_0.cuda()
        return (c_0,h_0)

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[1]
        x=self.audio_feature_block(x)
        x1=self.lstm1(x,self.init_hidden(batch_size,self.hidden_dim1))[0]
        x1=self.final_fc1(x1).reshape((batch_size*seq_length,self.output_dim1+self.output_dim2))

        bass_del=chord_limit.bass_slice_begin+12+1
        seventh_del=bass_del+chord_limit.seventh_limit+1
        ninth_del=seventh_del+chord_limit.ninth_limit+1
        eleventh_del=ninth_del+chord_limit.eleventh_limit+1
        thirteenth_del=eleventh_del+chord_limit.thirteenth_limit+1
        return x1[:,:chord_limit.bass_slice_begin],\
            x1[:,chord_limit.bass_slice_begin:bass_del],\
            x1[:,bass_del:seventh_del],\
            x1[:,seventh_del:ninth_del],\
            x1[:,ninth_del:eleventh_del],\
            x1[:,eleventh_del:thirteenth_del]

    def loss(self, x, y):
        output=self.feed(x)
        tag=y.view((-1,6))
        def conditional_classifier_loss(a,b):
            if((b<0).all()):
                return torch.tensor(0,device=b.device)
            loss=F.cross_entropy(a[b>=0],b[b>=0],reduce=True)
            #loss_term=self.loss_calc(a[b>=0],b[b>=0])
            return loss
        result=conditional_classifier_loss(output[0],tag[:,0])+\
        conditional_classifier_loss(output[1],tag[:,1]+1)+\
        conditional_classifier_loss(output[2],tag[:,2])+\
        conditional_classifier_loss(output[3],tag[:,3])+\
        conditional_classifier_loss(output[4],tag[:,4])+\
        conditional_classifier_loss(output[5],tag[:,5])
        return result

    def inference(self, x):
        seq_length=x.shape[0]
        output=self.feed(x[:,SHIFT_HIGH*SHIFT_STEP:SHIFT_HIGH*SHIFT_STEP+SPEC_DIM].view((1,seq_length,SPEC_DIM)))
        result_triad=F.softmax(output[0],dim=1).cpu().numpy()
        result_bass=F.softmax(output[1],dim=1).cpu().numpy()
        result_7=F.softmax(output[2],dim=1).cpu().numpy()
        result_9=F.softmax(output[3],dim=1).cpu().numpy()
        result_11=F.softmax(output[4],dim=1).cpu().numpy()
        result_13=F.softmax(output[5],dim=1).cpu().numpy()
        return result_triad,result_bass,result_7,result_9,result_11,result_13

mirex_chord_model=NetworkEnsemble(ChordNet,'jam_mirex2019_chordnet_v1_s%d.best',0,5,load_checkpoint=False,
                                  load_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','models','nn'))

class MirexChordProbability(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        if('source' in kwargs):
            music=entry.dict[kwargs['source']].get(entry)
        else:
            music=entry.music
        cqt=librosa.core.hybrid_cqt(music,
                                bins_per_octave=36,
                                fmin=librosa.note_to_hz('F#0'),
                                n_bins=288,
                                tuning=None,
                                hop_length=512).T
        cqt=abs(cqt).astype(np.float32)
        probs=mirex_chord_model.inference(cqt)
        return np.concatenate(probs,axis=1)

