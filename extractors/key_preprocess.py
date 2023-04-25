from mir.extractors import ExtractorBase
from mir import io
import numpy as np
from mir.music_base import get_scale_and_suffix

class KeyLabToScaleMode(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='key' if 'source' not in kwargs else kwargs['source']
        n_frame=entry.n_frame
        sr=entry.prop.sr
        hop_length=entry.prop.hop_length
        result=np.ones((n_frame),dtype=np.int8)*-1 # warning int8
        for key_info in entry.dict[proxy_name].get(entry):
            start_frame=max(int(np.floor(key_info[0]*sr/hop_length)),0)
            end_frame=min(int(np.floor(key_info[1]*sr/hop_length)),n_frame)
            key_scale,key_suffix=get_scale_and_suffix(key_info[2])
            result[start_frame:end_frame]=key_scale+\
                ['major','minor','dorian','mixolydian','lydian','phrygian','locrian'].index(key_suffix[1:])*12
        return result



