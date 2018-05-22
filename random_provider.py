import numpy as np


def random_provider(shape,raw,affin):
    x=int(np.random.random()*(np.shape(raw)[0]-shape[0]))
    y=int(np.random.random()*(np.shape(raw)[1]-shape[1]))
    z=int(np.random.random()*(np.shape(raw)[2]-shape[2]))
    return raw[x:x+shape[0],y:y+shape[1],z:z+shape[2]], affin[:,x:x+shape[0],y:y+shape[1],z:z+shape[2]]