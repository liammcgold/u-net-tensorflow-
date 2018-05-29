import numpy as np
import affinities as a


def drop_slice(raw, affins):

    index=int(np.random.rand()*(np.shape(raw)[0]))

    if(index==0 or index==np.shape(raw)[0]-1):
        index=int(.5*(np.shape(raw)[0]))

    raw[index]=raw[index+1]

    return raw, affins