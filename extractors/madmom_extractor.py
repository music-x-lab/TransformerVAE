from mir.extractors import ExtractorBase
from mir import io
from io_new.downbeat_io import DownbeatIO
from mir.data_file import FileProxy
from mir.settings import SONIC_ANNOTATOR_PATH
from mir.extractors.vamp_extractor import rewrite_extract_n3
from mir.cache import hasher
import librosa
import os
import numpy as np
import subprocess
from io_new.madmom_io import MadmomBeatProbIO
from mir import WORKING_PATH
from madmom.features import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor, CNNKeyRecognitionProcessor

from madmom.processors import ParallelProcessor, Processor, SequentialProcessor
from io_new.key_io import KeyIO
NUM_TO_ABS_SCALE=['A','Bb','B','C','Db','D','Eb','E','F','Gb','G','Ab']
KEY_LABEL=['%s\tmajor'%scale for scale in NUM_TO_ABS_SCALE]+['%s\tminor'%scale for scale in NUM_TO_ABS_SCALE]

MADMOM_SCRIPT_PATH=R'C:\Users\jjy3\AppData\Local\Programs\Python\Python36\Lib\site-packages\madmom-0.16.dev0-py3.6-win-amd64.egg\EGG-INFO\scripts'

class DBNDownBeatExtractor(ExtractorBase):


    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        source_name=kwargs['source'] if 'source' in kwargs else 'music'
        beats_per_bar=kwargs['beats_per_bar'] if 'beats_per_bar' in kwargs else [3, 4]
        max_bpm=kwargs['max_bpm'] if 'max_bpm' in kwargs else DBNDownBeatTrackingProcessor.MAX_BPM
        min_bpm=kwargs['min_bpm'] if 'min_bpm' in kwargs else DBNDownBeatTrackingProcessor.MIN_BPM
        print('DBNDownBeatExtractor working on entry '+entry.name)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            print('Error: not supported source type')
        filepath=entry.dict[source_name].filepath
        in_processor=RNNDownBeatProcessor()
        beat_only_mode = False
        if (-1 in beats_per_bar): # beat only
            beat_only_mode = True
            beats_per_bar = [1]
        beat_processor=DBNDownBeatTrackingProcessor(fps=100, online=False, beats_per_bar=beats_per_bar, max_bpm=max_bpm,
                                                    min_bpm=min_bpm)
        data=in_processor.process(filepath)
        if (beat_only_mode):
            data[:, 1] += data[:, 0]
            data[:, 0] = np.zeros_like(data[:, 0])
        data=beat_processor.process(data)
        return data


class DBNDecoder(ExtractorBase):

    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        print('DBNDecoder working on entry '+entry.name)
        fps=43.06640625
        if('fps' in kwargs):
            fps=kwargs['fps']
            print('Override fps by %.2f'%fps)
        beat_processor=DBNDownBeatTrackingProcessor(fps=fps,online=False,beats_per_bar=[3, 4],observation_lambda=16)
        data=entry.dict[kwargs['source']].get(entry)

        if(data.shape[1]==3):
            data=data[:,[2,1]]
        #PAD=64
        #raw_data=np.ones((data.shape[0]+PAD*2,2))*0.05
        #raw_data[PAD:-PAD,:]=
        data=beat_processor.process(data)
        return data

class TempoAwareDBNDecoder(ExtractorBase):

    def get_feature_class(self):
        return DownbeatIO

    def extract(self,entry,**kwargs):
        from extractors.hmm.hmm_wrapper import TempoAwareDownBeatTrackingProcessor
        print('DBNDecoder working on entry '+entry.name)
        fps=43.06640625
        if('fps' in kwargs):
            fps=kwargs['fps']
            print('Override fps by %.2f'%fps)
        beat_processor=TempoAwareDownBeatTrackingProcessor(fps=fps,online=False,beats_per_bar=[3, 4],observation_lambda=16)
        data=entry.dict[kwargs['source']].get(entry)
        data=beat_processor.process(data)
        return data


class DBNDownBeatProbability(ExtractorBase):

    def get_feature_class(self):
        return MadmomBeatProbIO

    def extract(self,entry,**kwargs):
        source_name=kwargs['source'] if 'source' in kwargs else 'music'
        print('DBNDownBeatProbability working on entry '+entry.name)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            filepath='temp/dbn_downbeat_probability_extractor_%s.wav'%hasher(entry.name)
            io.MusicIO().write(entry.dict[source_name].get(entry),filepath,entry)
        else:
            filepath=entry.dict[source_name].filepath
        in_processor=RNNDownBeatProcessor()
        data=in_processor.process(filepath)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            try:
                if(filepath.startswith('temp/')): # otherwise, bad things will happen
                    os.unlink(filepath)
            except:
                pass
        return data.astype(np.float16)

class MadmomSpectrogramProcessor(SequentialProcessor):

    def __init__(self,fps, **kwargs):
        # pylint: disable=unused-argument
        from functools import partial
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
        from madmom.ml.nn import NeuralNetworkEnsemble
        from madmom.models import DOWNBEATS_BLSTM

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        # instantiate a SequentialProcessor
        super(MadmomSpectrogramProcessor, self).__init__((pre_processor,))


class MadmomSpectrogram(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        source_name=kwargs['source'] if 'source' in kwargs else 'music'
        print('MadmomSpectrogram working on entry '+entry.name)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            filepath='temp/dbn_downbeat_probability_extractor_%s.wav'%hasher(entry.name)
            io.MusicIO().write(entry.dict[source_name].get(entry),filepath,entry)
        else:
            filepath=entry.dict[source_name].filepath
        in_processor=MadmomSpectrogramProcessor(fps=entry.prop.sr/entry.prop.hop_length)
        data=in_processor.process(filepath)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            try:
                if(filepath.startswith('temp/')): # otherwise, bad things will happen
                    os.unlink(filepath)
            except:
                pass
        return data

class MadmomKeyProbability(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        n_frame=entry.n_frame
        source_name=kwargs['source'] if 'source' in kwargs else 'music'
        print('DBNDownBeatProbability working on entry '+entry.name)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            filepath='temp/madmom_key_recognition_%s.wav'%hasher(entry.name)
            io.MusicIO().write(entry.dict[source_name].get(entry),filepath,entry)
        else:
            filepath=entry.dict[source_name].filepath
        in_processor=CNNKeyRecognitionProcessor()
        data=in_processor.process(filepath)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            try:
                if(filepath.startswith('temp/')): # otherwise, bad things will happen
                    os.unlink(filepath)
            except:
                pass
        return data.astype(np.float16).reshape((1,-1))+np.zeros((n_frame,1))

class MadmomKeyEstimation(ExtractorBase):

    def get_feature_class(self):
        return KeyIO

    def extract(self,entry,**kwargs):
        n_frame=entry.n_frame
        source_name=kwargs['source'] if 'source' in kwargs else 'music'
        print('DBNDownBeatProbability working on entry '+entry.name)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            filepath='temp/madmom_key_recognition_%s.wav'%hasher(entry.name)
            io.MusicIO().write(entry.dict[source_name].get(entry),filepath,entry)
        else:
            filepath=entry.dict[source_name].filepath
        in_processor=CNNKeyRecognitionProcessor()
        data=in_processor.process(filepath)
        if(not isinstance(entry.dict[source_name],FileProxy)):
            try:
                if(filepath.startswith('temp/')): # otherwise, bad things will happen
                    os.unlink(filepath)
            except:
                pass
        return [[0.,1.,KEY_LABEL[np.argmax(data.astype(np.float16).reshape((-1,)))]]]
