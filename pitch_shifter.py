import numpy as np
from music_base import VALID_MIDI_COUNT,MIDI_BASE
from basic_vae_blocks import WHOLE_CONDITION_DIM,WHOLE_X_DIM
from mir.nn.data_decorator import AbstractPitchShifter

def uncompress_data(data,shift=0,packed=False,erase_chord=False,fix_length_in_bar=-1):
    target_length=data.shape[0] if fix_length_in_bar==-1 else fix_length_in_bar*16
    if(packed):
        data=data.reshape((-16,data.shape[1]//16))
    data=data[:target_length]
    data_length=data.shape[0]
    p=0
    midi=data[:,p].astype(np.int32);p+=1
    midi[midi>0]+=shift-MIDI_BASE+1
    midi[midi<0]=0
    midi[midi>=VALID_MIDI_COUNT+1]=0
    onset=data[:,p].astype(np.int32);p+=1
    valid_chord_indices=data[:,p]>=0
    chord_root=(data[valid_chord_indices,p].astype(np.int32)+shift+12)%12;p+=1
    chord_map=np.roll(data[valid_chord_indices,p:p+12],shift+12,axis=1);p+=12
    chord_bass=(data[valid_chord_indices,p].astype(np.int32)+chord_root)%12;p+=1
    bar_pos=data[:,p].astype(np.int32);p+=1
    beat_pos=data[:,p].astype(np.int32);p+=1
    #if(p!=data.shape[1]):
    #    print('Warning: data shape inconsistent. Model expected %d, input has %d'%(p,data.shape[1]))
    result=np.zeros((target_length,WHOLE_CONDITION_DIM+WHOLE_X_DIM+1),dtype=np.float32)
    p=0
    midi_onset=midi
    midi_onset[midi_onset>0]+=1
    midi_onset[(onset==0)&(midi>0)]=1
    result[:data_length,p]=midi_onset;p+=1
    result[data_length:,p]=1.0 # no-midi-placeholder
    result[:data_length,p:p+VALID_MIDI_COUNT+2][np.arange(data_length),midi_onset]=1.0;p+=VALID_MIDI_COUNT+2
    if(erase_chord):
        p+=36
    else:
        result[:data_length,p:p+12][valid_chord_indices,chord_root]=1.0;p+=12
        result[:data_length,p:p+12][valid_chord_indices,:]=chord_map;p+=12
        result[:data_length,p:p+12][valid_chord_indices,chord_bass]=1.0;p+=12
    remain_counter=np.arange(target_length-data_length)
    result[data_length:,p:p+4][remain_counter,(remain_counter//4)%4]=1.0
    result[:data_length,p:p+4][np.arange(data_length),bar_pos]=1.0;p+=4
    result[data_length:,p:p+4][remain_counter,remain_counter%4]=1.0
    result[:data_length,p:p+4][np.arange(data_length),beat_pos]=1.0;p+=4
    assert(p==result.shape[1])
    return result


class GenPitchShifter(AbstractPitchShifter):

    def __init__(self,fix_bar_count=-1,erase_chord=False):
        self.erase_chord=erase_chord
        self.fix_bar_count=fix_bar_count
    #todo: chord N
    def pitch_shift(self,data,shift):
        return uncompress_data(data,shift,True,fix_length_in_bar=self.fix_bar_count,erase_chord=self.erase_chord)
