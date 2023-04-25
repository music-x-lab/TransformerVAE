import numpy as np
import subprocess
import os

def __dump_to_file(parameter, path, sr, mono):
    channels = 1 if mono else 2
    command = ['ffmpeg','-y']+parameter+[
        '-ar', str(sr),
        '-ac', str(channels),
        path]
    p = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    return p.returncode

def __receive_audio(parameter, sr, mono, normalize, in_type=np.int16, out_type=np.float32):
    # https://gist.github.com/kylemcdonald/85d70bf53e207bab3775
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = ['ffmpeg']+parameter+['-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    channels = 1 if mono else 2
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=4096)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        raise Exception('Audio load failed')
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio


def ffmpeg_load_audio(filename, sr, mono=True, normalize=True):
    return __receive_audio(['-i',filename],sr=sr,mono=mono,normalize=normalize)

def ffmpeg_load_video_track(filename,sr,track_id,mono=True, normalize=True):
    return __receive_audio(['-i',filename,
                            '-vn',
                            '-map','0:%d'%track_id],sr=sr,mono=mono,normalize=normalize)

def ffmpeg_convert_video_track(filename,sr,track_id,output_path,mono=True):
    return __dump_to_file(['-i',filename,
                            '-vn',
                            '-map','0:%d'%track_id],path=output_path,sr=sr,mono=mono)
if __name__ == '__main__':
    from mir import DataEntry
    from mir import io
    entry=DataEntry()
    entry.prop.set('sr',22050)
    audio=ffmpeg_load_video_track(
        R'D:\Dataset\karaoke-mini\拷贝MKV歌单，歌曲连接\拷贝MKV版本\何洁-请不要对我说Sorry(MTV)-国语-流行.MKV',
        sr=22050,track_id=2
    )
    entry.append_data(audio,io.MusicIO,'music')
    entry.visualize([])