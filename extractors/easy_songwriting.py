import numpy as np
from air_structure import AIRStructure
from array_to_midilab import air_to_midilab
from extractors.midilab_exporter import visualize_midilab

BASE_C=60

class SampleSong:
    '''
    SampleSong: a helper class to easily create your own song by writing text
    See the examples below for usage
    '''
    def __init__(self,name,bar_count,melody_text,chord_text,text_beat_division,bpm=120,num_beat_division=4,beat_per_bar=4,verbose_level=1):
        self.name=name
        self.bpm=bpm
        self.chord_text=chord_text
        self.beat_interval=60.0/bpm
        self.text_beat_division=text_beat_division
        self.air=AIRStructure(self.__create_beat_timing(bar_count,beat_per_bar),
                              num_beat_division=num_beat_division,verbose_level=verbose_level)
        if(melody_text!=''):
            self.air.append_melody(self.__extract_melody(melody_text))
        if(chord_text!=''):
            self.air.append_chord(self.__extract_chord(chord_text))

    def __create_beat_timing(self,bar_count,beat_per_bar):
        result=[]
        p=0
        for i in range(bar_count):
            for j in range(beat_per_bar):
                result.append([p*self.beat_interval,j+1])
                p+=1
        result.append([p*self.beat_interval,1])
        return np.array(result)

    def __extract_melody(self,melody_text):
        p=0
        base=BASE_C
        result=[]
        beat_pos=0
        while(p<len(melody_text)):
            token_type,token,p=self.__read_token(melody_text,p)
            if(token_type=='hint'):
                new_base=60
                if(token[0]=='-'):
                    new_base-=12
                new_base+=self.__to_scale(token,1)[0]
                base=new_base
            elif(token_type=='scale'):
                midi=(token-base)%12+base
                result.append([beat_pos,beat_pos+1,midi])
                beat_pos+=1
            elif(token_type=='sustain'):
                result[-1][1]+=1
                beat_pos+=1
            elif(token_type=='silence'):
                beat_pos+=1
            elif(token_type=='useless'):
                continue
            else:
                raise Exception('Bad token type in melody: %s'%token_type)
        return np.array([[x[0]*self.beat_interval/self.text_beat_division,
                          x[1]*self.beat_interval/self.text_beat_division,x[2]] for x in result])

    def __extract_chord(self,chord_text):
        p=0
        result=[]
        beat_pos=0
        while(p<len(chord_text)):
            token_type,token,p=self.__read_token(chord_text,p)
            if(token_type=='chord'):
                result.append([beat_pos,beat_pos+1,token])
                beat_pos+=1
            elif(token_type=='sustain'):
                result[-1][1]+=1
                beat_pos+=1
            elif(token_type=='useless'):
                continue
            else:
                raise Exception('Bad token type in chord: %s'%token_type)
        return [[x[0]*self.beat_interval,x[1]*self.beat_interval,x[2]] for x in result]


    def __to_scale(self,str,p=0):
        if(str[p]=='#'):
            scale,delta=self.__to_scale(str,p+1)
            return scale+1,delta
        elif(str[p]=='b'):
            scale,delta=self.__to_scale(str,p+1)
            return scale-1,delta
        return [0,2,4,5,7,9,11][int(str[p])-1],p+1


    def __read_token(self,str,p):
        if(p==len(str)):
            return 'EOS',None,p
        if(str[p]=='['):
            q=p
            while(str[q]!=']'):
                q+=1
            hint=str[p+1:q]
            return 'hint',hint,q+1
        if(str[p]=='-'):
            return 'sustain',None,p+1
        if(str[p]=='0'):
            return 'silence',None,p+1
        if((str[p]>='1' and str[p]<='7') or str[p]=='#' or str[p]=='b'):
            scale,q=self.__to_scale(str,p)
            return 'scale',scale,q
        if((str[p]>='A' and str[p]<='G') or str[p]=='N' or str[p]=='X'):
            q=p
            while(q<len(str) and str[q] not in '[-|,'):
                q+=1
            chord=str[p:q]
            return 'chord',chord,q
        if(str[p] in '|,\n\t '):
            return 'useless',None,p+1
        raise Exception('Unknown token %s'%str[p])

    def visualize(self):
        visualize_midilab(air_to_midilab(self.air,melody=True,chord=False),air_to_midilab(self.air,melody=False,chord=True))

