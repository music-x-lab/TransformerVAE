import numpy as np
from pitch_shifter import uncompress_data


TRAIN_BAR_COUNT=8

def full_eval(model,entries):
    def average_kld(z_mu,z_sigma):
        return (-0.5*np.mean(1+z_sigma-z_mu**2-np.exp(z_sigma)))
    def accuracy(data,recons):
        return (data[:,0]==np.argmax(recons,axis=1)).sum()/len(data)
    print('Eval: %s'%model.save_name)
    acc_sampling=[]
    acc_teacher_forcing=[]
    kld=[]
    for i,entry in enumerate(entries):
        if(i%100==0):
            print('%d / %d'%(i,len(entries)),flush=True)
        fix_length_in_bar=TRAIN_BAR_COUNT
        air=entry.air
        timing,all_data=air.export_to_array()
        data=uncompress_data(all_data,fix_length_in_bar=fix_length_in_bar,erase_chord=True)
        z=model.inference_function('inference_encode',data)
        mu,logvar=model.inference_function('collect_statistics',data)
        recons_sampling=model.inference_function('inference_decode',z,data)
        recons_teacher_forcing=model.inference_function('inference_decode_cheat',z,data)
        entry_acc_sampling=accuracy(data,recons_sampling)
        entry_acc_teacher_forcing=accuracy(data,recons_teacher_forcing)
        entry_kld=average_kld(mu,logvar)
        #print(data.shape,recons_sampling.shape,recons_teacher_forcing.shape)
        #print(acc_sampling,acc_teacher_forcing,entry_kld)
        acc_sampling.append(entry_acc_sampling)
        acc_teacher_forcing.append(entry_acc_teacher_forcing)
        kld.append(entry_kld)
    print('[RESULT]')
    print('ACC SAMPLING:\t%.4f'%np.mean(acc_sampling))
    print('ACC TEACHER F.:\t%.4f'%np.mean(acc_teacher_forcing))
    print('KLD:\t%.4f'%np.mean(kld))
    print('',flush=True)