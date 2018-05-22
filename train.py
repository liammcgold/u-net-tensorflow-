import create_model
import numpy as np
import h5py
import tensorflow as tf
import random_provider as rp

f = h5py.File("sample_A_20160501.hdf", "r")
raw_data=f["volumes"]["raw"].value

affinities=np.load("affinities.npy")

raw_data=np.asarray(raw_data)







init, opt, raw_input, target =create_model.create_model((None,1,16,128, 128),(None,3,16, 128, 128))

with tf.Session() as sess:
    sess.run(init)
    n=0
    for i in range(100):
        raw_data_samp, affinities_samp=rp.random_provider((16,128,128),raw_data,affinities)
        raw_in=np.zeros((1,1,16,128,128))
        affin_in=np.zeros((1,3,16,128,128))
        raw_in[0]=raw_data_samp
        affin_in[0]=affinities_samp
        print(np.shape(raw_data_samp))
        sess.run([opt],feed_dict={raw_input:raw_in,target:affin_in})