import numpy as np
import affinities as a

def mirror(raw, affins):
    raw=np.flip(raw,2)
    affins=a.get_affins(raw)
    return raw, affins