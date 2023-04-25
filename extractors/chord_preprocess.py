from mir.extractors import ExtractorBase
from mir import io
import numpy as np
import complex_chord

class ChordLabToFramedComplexChord(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        proxy_name='chord' if 'source' not in kwargs else kwargs['source']
        n_frame=entry.n_frame
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        tags = np.ones((n_frame,6))*-2
        for tokens in entry.dict[proxy_name].get(entry):
            begin=int(round(float(tokens[0])*sr/win_shift))
            end = int(round(float(tokens[1])*sr/win_shift))
            if (end > n_frame):
                end = n_frame
            if(begin<0):
                begin=0
            tags[begin:end,:]=complex_chord.Chord(tokens[2]).to_numpy().reshape((1,6))
        return tags


