import numpy as np 

def get_next_done(terminated):
    return 1 if terminated else 0

get_next_done_vectorized = np.vectorize(get_next_done)
        
