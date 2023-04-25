import torch.nn as nn
import torch.nn.functional as F
from mir.nn.train import NetworkBehavior,NetworkInterface
import torch
import numpy as np
from mir.extractors import ExtractorBase
from io_new.downbeat_io import DownbeatIO
import librosa
from madmom.features import DBNDownBeatTrackingProcessor
import os

LSTM_TRAIN_LENGTH=1000
BEAT_CLASSES=3
SPEC_DIM=256
REWEIGHT_NO_BEAT=1.0

class DilatedBlockV2(nn.Module):
    def __init__(self,channel):
        super(DilatedBlockV2, self).__init__()
        self.conv1=nn.Conv2d(channel,channel,(3,3),padding=(1,1),dilation=(1,1))
        self.conv2=nn.Conv2d(channel,channel//2,(3,3),padding=(2,1),dilation=(2,1))
        self.conv3=nn.Conv2d(channel,channel//2,(3,3),padding=(4,1),dilation=(4,1))
        self.conv4=nn.Conv2d(channel,channel//2,(3,3),padding=(8,1),dilation=(8,1))
        self.conv5=nn.Conv2d(channel,channel//2,(3,3),padding=(16,1),dilation=(16,1))
        self.conv6=nn.Conv2d(channel,channel//2,(3,3),padding=(32,1),dilation=(32,1))
        self.conv7=nn.Conv2d(channel,channel//2,(3,3),padding=(64,1),dilation=(64,1))
        self.conv8=nn.Conv2d(channel,channel//2,(3,3),padding=(128,1),dilation=(128,1))
        self.conv9=nn.Conv2d(channel,channel//2,(3,3),padding=(256,1),dilation=(256,1))
        self.conv2b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv3b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv4b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv5b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv6b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv7b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv8b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))
        self.conv9b=nn.Conv2d(channel,channel//2,(3,3),padding=(1,1))


    def forward(self, *x):
        if(isinstance(x,tuple)):
            x=x[0]
        x=F.selu(self.conv1(x))+x
        x=F.selu(torch.cat((self.conv2(x),self.conv2b(x)),1))+x
        x=F.selu(torch.cat((self.conv3(x),self.conv3b(x)),1))+x
        x=F.selu(torch.cat((self.conv4(x),self.conv4b(x)),1))+x
        x=F.selu(torch.cat((self.conv5(x),self.conv5b(x)),1))+x
        x=F.selu(torch.cat((self.conv6(x),self.conv6b(x)),1))+x
        x=F.selu(torch.cat((self.conv7(x),self.conv7b(x)),1))+x
        x=F.selu(torch.cat((self.conv8(x),self.conv8b(x)),1))+x
        x=F.selu(torch.cat((self.conv9(x),self.conv9b(x)),1))+x
        return x

class RCNNClassifierV4(NetworkBehavior):

    def __init__(self,half_hidden_dim):
        super(RCNNClassifierV4, self).__init__()
        self.hidden_dim=half_hidden_dim*2


        self.conv1=nn.Conv2d(1,24,(3,3),padding=(1,1))
        self.dilated1=DilatedBlockV2(24)
        self.pool1=nn.MaxPool2d((1,3))
        self.conv2=nn.Conv2d(24,48,(3,3),padding=(1,1))
        self.dilated2=DilatedBlockV2(48)
        self.pool2=nn.MaxPool2d((1,2))
        self.conv3=nn.Conv2d(48,80,(3,3),padding=(1,1))
        self.dilated3=DilatedBlockV2(80)
        self.pool3=nn.MaxPool2d((1,3))
        # Todo: correct?
        self.input_size=14*80
        self.lstm1=nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_dim//2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2=nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim//2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.final_fc=nn.Conv1d(self.hidden_dim,BEAT_CLASSES,1)
        self.beat_class_weight=torch.ones((BEAT_CLASSES))
        self.beat_class_weight[0]=REWEIGHT_NO_BEAT
        self.half_beat_class_weight=torch.ones((2))
        self.half_beat_class_weight[0]=REWEIGHT_NO_BEAT
        if(self.use_gpu):
            self.beat_class_weight=self.beat_class_weight.cuda()
            self.half_beat_class_weight=self.half_beat_class_weight.cuda()

    def init_hidden(self,batch_size,x):
        c_0=torch.zeros(2,batch_size,self.hidden_dim//2,device=x.device)
        h_0=torch.zeros(2,batch_size,self.hidden_dim//2,device=x.device)
        return (c_0,h_0)

    def forward(self, x):
        batch_size=x.shape[0]
        seq_length=x.shape[2]
        x=self.pool1(self.dilated1(F.selu(self.conv1(x))))
        x=self.pool2(self.dilated2(F.selu(self.conv2(x))))
        x=self.pool3(self.dilated3(F.selu(self.conv3(x))))
        x=x.transpose(1,2).contiguous().view((batch_size,seq_length,self.input_size))

        x=self.lstm1(x,self.init_hidden(batch_size,x))[0]
        x=self.lstm2(x,self.init_hidden(batch_size,x))[0]
        y=self.final_fc(x.reshape((-1,self.hidden_dim,1))).reshape((-1,seq_length,BEAT_CLASSES))
        return y

    def loss(self, x, z):
        seq_length=x.shape[1]
        beat_output=self.feed(x.view((-1,1,seq_length,SPEC_DIM)))
        z=z.view((-1,2))
        z[z[:,0]>=2,0]=2
        valid_metres=(z[:,1]>=3)&(z[:,1]<=4)
        valid_indices=(z[:,0]>=0)&valid_metres
        half_valid_indices=(z[:,0]<=-2)&valid_metres
        beat_result=F.log_softmax(beat_output.view((-1,BEAT_CLASSES)), 1)
        loss=torch.zeros((1),device=x.device)
        if(torch.sum(valid_indices)>0):
            loss+=F.nll_loss(beat_result[valid_indices,:],z[valid_indices,0],reduce=False,weight=self.beat_class_weight).sum()
        if(torch.sum(half_valid_indices)>0):
            beat_seg=beat_result[half_valid_indices,:]
            beat_result_2=torch.zeros((beat_seg.shape[0],2),device=beat_seg.device)
            beat_result_2[:,0]=beat_seg[:,0]
            beat_result_2[:,1]=torch.log(F.softmax(beat_output.view((-1,BEAT_CLASSES))[half_valid_indices,:], 1)[:,1:].sum(1))
            loss+=F.nll_loss(beat_result_2,1+0*z[half_valid_indices,0],reduce=False,weight=self.half_beat_class_weight).sum()
        return loss/z.shape[0]

    def inference(self, x):
        with torch.no_grad():
            beat_output=self(x[None,None,:,:])

        return F.softmax(beat_output.view((-1,BEAT_CLASSES)),dim=1).cpu().detach().numpy()

osu_beat_model=NetworkInterface(RCNNClassifierV4(256),'osu_binary_beat_rcnnv4.2_s2.cp',load_checkpoint=False,
                                load_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','models','nn'))


def dbn_decode(beat_prob,fps,offset):
    beat_processor=DBNDownBeatTrackingProcessor(fps=fps,online=False,beats_per_bar=[3, 4],observation_lambda=16)
    if(beat_prob.shape[1]==3):
        data=beat_prob[:,[2,1]]
    else:
        data=beat_prob
    result=beat_processor.process(data)
    result[:,0]+=offset
    return result

class OsuBeatExtractorV1(ExtractorBase):

    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        source=kwargs['source'] if 'source' in kwargs else 'music'
        stretch=kwargs['stretch'] if 'stretch' in kwargs else 1.0
        wave=entry.dict[source].get(entry)
        sr=entry.prop.sr
        hop_length=256
        fps=86.1328125
        offset=0.04
        if(stretch!=1.0):
            if(stretch>0.0):
                wave=librosa.effects.time_stretch(wave,stretch)
            else:
                wave=librosa.core.resample(wave,sr,int(np.round(sr/-stretch)))

        cqt_stretch=librosa.core.hybrid_cqt(wave,
                                bins_per_octave=36,
                                n_bins=256,
                                fmin=librosa.midi_to_hz(30),
                                hop_length=hop_length).T
        beat_prob=osu_beat_model.inference(cqt_stretch)
        return dbn_decode(beat_prob,fps=fps/np.abs(stretch),offset=offset)

