from io_new.midilab_io import MidiLabIO
import numpy as np
from music_base import MIDI_BASE,VALID_MIDI_COUNT
import mir_eval.chord

CHORD_BASS_START=36+7-12
CHORD_MAIN_START=36+7


def activation_to_midilab(activation,timing):
    last_pitch=-1
    last_onset=-1
    result=[]
    n_frame=len(activation)
    for i in range(n_frame):
        best_choice=np.argmax(activation[i,:])
        if(best_choice==0 or best_choice>1):
            if(last_pitch>=0):
                result.append([last_onset,i-1,last_pitch+MIDI_BASE])
            if(best_choice>1):
                last_onset=i
                last_pitch=best_choice-2
            else:
                last_pitch=-1
    if(last_pitch>=0):
        result.append([last_onset,n_frame-1,last_pitch+MIDI_BASE])
    return [[timing[x[0]][0],timing[x[1]][1]-1e-3,x[2]] for x in result]


def air_to_activation(air):
    timing,array=air.export_to_array()
    midi=array[:,0].astype(np.int32)
    onset=array[:,1].astype(np.int32)
    midi[midi>0]+=-MIDI_BASE+1
    midi[midi<0]=0
    midi[midi>=VALID_MIDI_COUNT+1]=0

    midi_onset=midi
    midi_onset[midi_onset>0]+=1
    midi_onset[(onset==0)&(midi>0)]=1

    result=np.zeros((array.shape[0],VALID_MIDI_COUNT+2))
    result[np.arange(result.shape[0]),midi_onset]=1
    return result,timing

def air_to_midilab(air,melody=True,chord=True):
    result=[]
    if(melody):
        activation,timing=air_to_activation(air)
        result+=activation_to_midilab(activation,timing)
    if(chord):
        result+=air_to_chord_midilab(air)
    return result


def air_to_chord_midilab(air):
    timing,array=air.export_to_array(export_all=True)
    last_chord_onset=-1
    last_chord_notes=[]
    result=[]
    for i in range(len(array)+1):
        if(i==len(array) or air.is_downbeat[i] or i==0 or air.chord[i]!=air.chord[i-1]):
            if(last_chord_onset>=0):
                result+=[[last_chord_onset,i-1,note] for note in last_chord_notes]
                last_chord_onset=-1
            if(i<len(array)):
                root,chroma,bass=mir_eval.chord.encode(air.chord[i] if air.chord[i]!='' else 'N')
                last_chord_notes=chord_array_to_midi_notes(root,(root+bass)%12,chroma)
                last_chord_onset=i
    return [[timing[x[0]][0],timing[x[1]][1]-1e-3,x[2]] for x in result]

def chord_array_to_midi_notes(root,bass,relative_map):
    if(bass<0):
        return []
    result=[(bass-CHORD_BASS_START)%12+CHORD_BASS_START]
    for i in range(0,12):
        if(relative_map[i]>0):
            result.append((i+root-CHORD_MAIN_START)%12+CHORD_MAIN_START)
    return result

def midilab_to_trinket(midilab,bpm,accidental='#'):
    num_2_letter={'#':['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'],
                  'b':['C','D-','D','E-','E','F','G-','G','A-','A','B-','B']}

    num_2_octave=['XXXX','XXX','XX','X','x','x\'','x\'\'','x\'\'\'']
    frame_interval=60./bpm/4
    aligned_notes=[]
    length=0
    for note in midilab:
        note_start=int(np.round(note[0]/frame_interval))
        note_end=int(np.round(note[1]/frame_interval))
        #print(note_start,note_end,note[2])
        if(note_end==note_start):
            continue
        length=max(length,note_end)
        aligned_notes.append([note_start,note_end,note[2]])
    aligned_notes.sort(key=lambda note:note[0])
    total_length=((length-1)//16+1)*16
    result=[]
    p=0
    def single_pitch_to_text(pitch):
        if(pitch==-1):
            note_text='r'
        else:
            letter=num_2_letter[accidental][pitch%12]
            octave=num_2_octave[pitch//12].replace('X',letter[0]).replace('x',letter[0].lower())
            note_text=octave+letter[1:]
        return note_text
    def pitches_to_text(pitches):
        if(len(pitches)>1):
            return '<'+(' '.join(single_pitch_to_text(pitch) for pitch in pitches)) + '>'
        else:
            return single_pitch_to_text(pitches[0])
    def append_note(pitches,length):
        pitch_text=pitches_to_text(pitches)
        if(length%3==0):
            text='%s%d.'%(pitch_text,16//(length//3*2))
        else:
            text='%s%d'%(pitch_text,16//length)
        result.append(text)


    def append_interval(p,pitches,length):

        lowbit=p&-p
        if(lowbit>16 or lowbit==0):
            lowbit=16
        if(lowbit%length==0):
            append_note(pitches,length)
            p+=length
        elif(pitches[0]!=-1 and length*4==lowbit*3):  # dotted note
            append_note(pitches,length)
            p+=length
        else:
            for l in range(length,0,-1):
                if(lowbit%l==0):
                    append_note(pitches,l)
                    p+=l
                    if(pitches[0]!=-1):
                        result.append('~')
                    p=append_interval(p,pitches,length-l)
                    break
        return p
    i=0
    while(i<len(aligned_notes)):
        note=aligned_notes[i]
        if(note[0]>p):
            p=append_interval(p,[-1],note[0]-p)
        k=i
        while(k<len(aligned_notes)+1):
            if(k<len(aligned_notes) and aligned_notes[k][0]==note[0]):
                assert(aligned_notes[k][1]==note[1])
                k+=1
            else:
                p=append_interval(p,[aligned_notes[j][2] for j in range(i,k)],note[1]-note[0])
                i=k-1
                break
        i+=1
    if(p<total_length):
        p=append_interval(p,[-1],total_length-p)
    assert(p==total_length)
    return ' '.join(result)


if __name__ == '__main__':
    # perform some tests
    midilab=[
        [1.0,1.25,60],
        [1.25,1.75,62],
        [1.75,2.25,59],
        [7.0,11.0,62],
        [7.0,11.0,63-24],
        [7.0,11.0,63-48],
    ]
    print(midilab_to_trinket(midilab,60))