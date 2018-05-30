import numpy as np
import affinities as a


def rotate(raw, affins):
    raw_temp=np.swapaxes(raw,-2,-1)
    affins_temp=a.get_affins(raw_temp)
    return raw_temp, affins_temp