import numpy as np
def analyze_double_speed_error(beat,window_size=8,const_speed_ratio_threshold=1.1,double_speed_threshold=1.8):
    log_threshold=np.log(const_speed_ratio_threshold)
    n_beat=len(beat)
    dist=np.array([beat[i+1][0]-beat[i][0] for i in range(n_beat-1)])
    min_dist=np.inf
    max_dist=0
    for i in range(n_beat-window_size-1):
        windowed_dist=dist[i:i+window_size]
        running_mean=np.mean(windowed_dist)
        deviation=np.abs(np.log(windowed_dist)-np.log(running_mean)).max()
        if(deviation<=log_threshold):
            min_dist=min(min_dist,running_mean)
            max_dist=max(max_dist,running_mean)
    speed_change=max_dist/min_dist
    print(speed_change)
    if(speed_change>double_speed_threshold):
        return True
    else:
        return False
