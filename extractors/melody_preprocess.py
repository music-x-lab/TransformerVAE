from mir.extractors import ExtractorBase
from mir import io
import numpy as np

class MelodyLabToCompressedChroma(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='melody' if 'source' not in kwargs else kwargs['source']
        beat=entry.beat
        beat_start=beat[0][0]
        beat_end=beat[-1][0]
        n_frame=entry.n_frame
        sr=entry.prop.sr
        hop_length=entry.prop.hop_length
        result=np.zeros((n_frame),dtype=np.int16)
        result[:max(int(np.floor(beat_start*sr/hop_length)),0)]=-1
        result[min(int(np.floor(beat_end*sr/hop_length)),n_frame):]=-1
        for note_info in entry.dict[proxy_name].get(entry):
            start_frame=max(int(np.floor(note_info[0]*sr/hop_length)),0)
            end_frame=min(int(np.floor(note_info[1]*sr/hop_length)),n_frame)
            scale=int(note_info[2])%12
            result[start_frame:end_frame]=np.maximum(result[start_frame:end_frame],0)
            result[start_frame:end_frame]=np.bitwise_or(result[start_frame:end_frame],1<<scale)
        return result


class FramedMelodyAndOnset(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='melody' if 'source' not in kwargs else kwargs['source']
        beat=entry.beat
        beat_start=beat[0][0]
        beat_end=beat[-1][0]
        n_frame=entry.n_frame
        sr=entry.prop.sr
        hop_length=entry.prop.hop_length
        result=np.zeros((n_frame,2),dtype=np.int16)
        result[:max(int(np.floor(beat_start*sr/hop_length)),0),:]=-1
        result[min(int(np.floor(beat_end*sr/hop_length)),n_frame):,:]=-1
        for note_info in entry.dict[proxy_name].get(entry):
            start_frame=max(int(np.floor(note_info[0]*sr/hop_length)),0)
            end_frame=min(int(np.floor(note_info[1]*sr/hop_length)),n_frame)
            result[start_frame:end_frame,:]=np.maximum(result[start_frame:end_frame],0)
            result[start_frame:end_frame,0]=note_info[2]
            result[start_frame:start_frame+1,1]=1
        return result

class MelodyLabToOnset(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='melody' if 'source' not in kwargs else kwargs['source']
        beat=entry.beat
        beat_start=beat[0][0]
        beat_end=beat[-1][0]
        n_frame=entry.n_frame
        sr=entry.prop.sr
        hop_length=entry.prop.hop_length
        result=np.zeros((n_frame),dtype=np.int16)
        result[:max(int(np.floor(beat_start*sr/hop_length)),0)]=-1
        result[min(int(np.floor(beat_end*sr/hop_length)),n_frame):]=-1
        for note_info in entry.dict[proxy_name].get(entry):
            start_frame=max(int(np.floor(note_info[0]*sr/hop_length)),0)
            if(start_frame<n_frame):
                result[start_frame]=1.0
        return result

class SubbeatLevelMelody(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='melody' if 'source' not in kwargs else kwargs['source']
        sr=entry.prop.sr
        hop_length=entry.prop.hop_length
        subbeat=entry.subbeat
        n_frame=len(subbeat)
        pos_valid=np.arange(n_frame)[subbeat>0]
        if(len(pos_valid)<=1):
            return -np.ones((n_frame),dtype=np.int16)
        result=np.zeros((n_frame),dtype=np.int16)
        result[:pos_valid[0]]=-1
        result[pos_valid[-1]:]=-1
        gap_pos=(pos_valid[1:]+pos_valid[:-1])//2
        for note_info in entry.dict[proxy_name].get(entry):
            start_frame=int(np.round(note_info[0]*sr/hop_length))
            end_frame=int(np.round(note_info[1]*sr/hop_length))
            gap_start=np.searchsorted(gap_pos,start_frame)
            gap_end=np.searchsorted(gap_pos,end_frame)
            if(gap_end>gap_start):
                if(start_frame+(pos_valid[1]-pos_valid[0])/2>=pos_valid[0]): # so that the first note is not too early
                    result[pos_valid[gap_start]]=int(note_info[2])%12+13
                result[pos_valid[gap_start]+1:pos_valid[gap_end]]=int(note_info[2])%12+1
        return result



