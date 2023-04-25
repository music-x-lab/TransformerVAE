from mir.nn.train import NetworkInterface
import numpy as np
class NetworkEnsemble():

    def __init__(self,net_class,net_names,idx_start,idx_end,*args,load_checkpoint=True,load_path='cache_data',**kwargs):
        self.net_class=net_class
        self.models=[NetworkInterface(net_class(*args,**kwargs),net_names%idx,load_checkpoint=load_checkpoint,load_path=load_path) for idx in range(idx_start,idx_end)]
        self.save_name=net_names
        for model in self.models:
            model.net.use_gpu=False

    def inference(self,*args,n_jobs=1):
        from joblib import Parallel,delayed
        result=Parallel(n_jobs=n_jobs,verbose=10)(delayed(NetworkInterface.inference)(model,*args) for model in self.models)
        if(isinstance(result[0],list) or isinstance(result[0],tuple)):
            return [np.mean([tokens[i] for tokens in result],axis=0) for i in range(len(result[0]))]
        else:
            return np.mean(result,axis=0)
