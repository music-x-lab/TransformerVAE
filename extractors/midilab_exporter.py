import numpy as np
from mir import DataEntry
from settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
from io_new.midilab_io import MidiLabIO
from mir.extractors.misc import BlankMusic
import pretty_midi
from mir.cache import mkdir_for_file

def visualize_midilab(*args):
    entry=DataEntry()
    entry.prop.set('sr',DEFAULT_SR)
    entry.prop.set('hop_length',DEFAULT_HOP_LENGTH)
    max_time=0.0
    p=0
    visualize_list=[]
    for midilab in args:
        proxy_name='midilab_%d'%p
        visualize_list.append(proxy_name)
        entry.append_data(midilab,MidiLabIO,proxy_name)
        if(len(midilab)>0):
            max_time=max(max_time,np.max(np.array(midilab)[:,:2]))
        p+=1
    entry.append_extractor(BlankMusic,'music',time=max_time+10.0)
    export_midi('temp/export.mid',*args)
    entry.visualize(visualize_list)

def midilab_connect(*args,interval_time=0.0):
    max_time=0.0
    for midilab in args:
        if(len(midilab)>0):
            max_time=max(max_time,np.max(np.array(midilab)[:,:2]))
    result_midilab=[]
    p=0
    step_time=max_time+interval_time
    for midilab in args:
        if(len(midilab)>0):
            result_midilab+=[[p*step_time+x[0],p*step_time+x[1],x[2]] for x in list(midilab)]
        p+=1
    return result_midilab

def export_midi(file_path,*args):
    midi=pretty_midi.PrettyMIDI()
    piano_program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    for midilab in args:
        piano=pretty_midi.Instrument(program=piano_program)
        for note in midilab:
            assert(note[1]>note[0]+1e-6)
            midi_note=pretty_midi.Note(velocity=100,pitch=note[2],start=note[0],end=note[1])
            piano.notes.append(midi_note)
        midi.instruments.append(piano)
    mkdir_for_file(file_path)
    midi.write(file_path)