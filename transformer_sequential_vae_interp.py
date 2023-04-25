
from transformer_sequential_vae import NetworkInterface,TransformerSequentialVAE,TRAIN_BAR_COUNT
from air_structure import AIRStructure
from array_to_midilab import activation_to_midilab,air_to_midilab,midilab_to_trinket
from pitch_shifter import uncompress_data
from extractors.midilab_exporter import visualize_midilab,midilab_connect,export_midi
import numpy as np
import os
from sample_songs import song_8_list
from basic_vae_blocks import interp_path
import rowan


def get_song_latent_code(model,air:AIRStructure):
    timing,data=air.export_to_array()
    data=uncompress_data(data,erase_chord='no_chord' in model.save_name)
    assert(len(data)==TRAIN_BAR_COUNT*16)
    return model.inference_function('inference_encode',data)

def latent_code_to_activation(model,z,air:AIRStructure):
    timing,data=air.export_to_array()
    data=uncompress_data(data,erase_chord='no_chord' in model.save_name)
    assert(len(data)==TRAIN_BAR_COUNT*16)
    return model.inference_function('inference_decode',z,data)

def latent_code_to_midilab(model,z,air:AIRStructure):
    timing,data=air.export_to_array()
    data=uncompress_data(data,erase_chord='no_chord' in model.save_name)
    assert(len(data)==TRAIN_BAR_COUNT*16)
    activation=model.inference_function('inference_decode',z,data)
    return activation_to_midilab(activation,timing)

def z_to_midilab(model,z,air:AIRStructure):
    timing,data=air.export_to_array()
    data=uncompress_data(data)
    assert(len(data)==TRAIN_BAR_COUNT*16)
    midi_result=model.inference_function('inference_decode',z,data)
    return activation_to_midilab(midi_result,timing)

def add_octave(model,song,visualize=True):

    chord_midilab=air_to_midilab(song.air,melody=False,chord=True)
    midi_chord=[chord_midilab,[],chord_midilab,[],chord_midilab]
    midi_melody=[air_to_midilab(song.air,melody=True,chord=False),[],[],[],[]]
    z=get_song_latent_code(model,song.air)
    midi_melody[3]=latent_code_to_midilab(model,z,song.air)
    midi_melody[6]=latent_code_to_midilab(model,np.concatenate((z2[:,0:1,:],z1[:,1:,:]),axis=1),song1.air)
    midi_melody[7]=latent_code_to_midilab(model,np.concatenate((z1[:,0:1,:],z2[:,1:,:]),axis=1),song2.air)
    file_path=os.path.join('output/transformer_sequential/%s/swap_first'%model.save_name,
                           '%s-%s.mid'%(song1.name,song2.name))
    midi_chord=midilab_connect(*midi_chord)
    midi_melody=midilab_connect(*midi_melody)
    export_midi(file_path,midi_melody,midi_chord)
    if(visualize):
        visualize_midilab(midi_melody,midi_chord)

def swap_first(model,song1,song2,visualize=True):

    chord_midilab1=air_to_midilab(song1.air,melody=False,chord=False)
    chord_midilab2=air_to_midilab(song2.air,melody=False,chord=False)
    midi_chord=[chord_midilab1,chord_midilab2,[],chord_midilab1,chord_midilab2,[],chord_midilab1,chord_midilab2]
    midi_melody=[air_to_midilab(song1.air,melody=True,chord=False),
                 air_to_midilab(song2.air,melody=True,chord=False),[],[],[],[],[],[]]
    z1=get_song_latent_code(model,song1.air)
    z2=get_song_latent_code(model,song2.air)
    midi_melody[3]=latent_code_to_midilab(model,z1,song1.air)
    midi_melody[4]=latent_code_to_midilab(model,z2,song2.air)
    midi_melody[6]=latent_code_to_midilab(model,np.concatenate((z2[:,0:1,:],z1[:,1:,:]),axis=1),song1.air)
    midi_melody[7]=latent_code_to_midilab(model,np.concatenate((z1[:,0:1,:],z2[:,1:,:]),axis=1),song2.air)
    file_path=os.path.join('output/transformer_sequential/%s/swap_first'%model.save_name,
                           '%s-%s.mid'%(song1.name,song2.name))
    midi_chord=midilab_connect(*midi_chord)
    midi_melody=midilab_connect(*midi_melody)
    export_midi(file_path,midi_melody,midi_chord)
    if(visualize):
        visualize_midilab(midi_melody,midi_chord)

def swap_first_two(model,song1,song2,visualize=True):

    use_chord='no_chord' not in model.save_name
    chord_midilab1=air_to_midilab(song1.air,melody=False,chord=False)
    chord_midilab2=air_to_midilab(song2.air,melody=False,chord=False)
    midi_chord=[chord_midilab1,chord_midilab2,[],chord_midilab1,chord_midilab2,[],chord_midilab1,chord_midilab2] if use_chord else [[]]*8
    midi_melody=[air_to_midilab(song1.air,melody=True,chord=False),
                 air_to_midilab(song2.air,melody=True,chord=False),[],[],[],[],[],[]]
    z1=get_song_latent_code(model,song1.air)
    z2=get_song_latent_code(model,song2.air)
    midi_melody[3]=latent_code_to_midilab(model,z1,song1.air)
    midi_melody[4]=latent_code_to_midilab(model,z2,song2.air)
    midi_melody[6]=latent_code_to_midilab(model,np.concatenate((z2[:,0:2,:],z1[:,2:,:]),axis=1),song1.air)
    midi_melody[7]=latent_code_to_midilab(model,np.concatenate((z1[:,0:2,:],z2[:,2:,:]),axis=1),song2.air)
    file_path=os.path.join('output/transformer_sequential/%s/swap_first_two'%model.save_name,
                           '%s-%s.mid'%(song1.name,song2.name))
    midi_chord=midilab_connect(*midi_chord)
    midi_melody=midilab_connect(*midi_melody)
    export_midi(file_path,midi_melody,midi_chord)
    if(visualize):
        visualize_midilab(midi_melody,midi_chord)

def visualize_attn_matrix(model,song):
    timing,data=song.air.export_to_array()
    data=uncompress_data(data,erase_chord='no_chord' in model.save_name)
    assert(len(data)==TRAIN_BAR_COUNT*16)
    z,attn=model.inference_function('inference_encode_return_attn',data)
    import matplotlib.pyplot as plt
    id=0
    fig, axes = plt.subplots(nrows=attn.shape[0], ncols=attn.shape[1])
    for i in range(attn.shape[0]):
        for j in range(attn.shape[1]):
            ax=plt.subplot(attn.shape[0],attn.shape[1],id+1)
            if(i==0):
                ax.set_title('Head %d'%j)
            if(j==0):
                ax.set_ylabel('Encoder Layer %d'%i)
            im=ax.imshow(attn[i,j,:,:])
            id+=1
    fig.subplots_adjust(right=0.75)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    fig.suptitle(song.name)
    plt.show()


if __name__ == '__main__':
    model1=NetworkInterface(TransformerSequentialVAE(256,64,0.5,1,1,1,pretrain=True),'transformer_sequential_vae_no_chord_v2.0_m111_3_layer_kl0.500000_s0',load_checkpoint=True)
    from sample_songs import song_8_list_new,test_fast_8,test2_fast_8,test3_fast_8
    song_list=song_8_list_new
    #for song in song_list:
    #    visualize_attn_matrix(model1,song)

    for model in [model1]:
        for i in range(len(song_list)):
            for j in range(i+1,len(song_list)):
                swap_first_two(model,song_list[i],song_list[j],visualize=False)
                swap_first(model,song_list[i],song_list[j],visualize=False)