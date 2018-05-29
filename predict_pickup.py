import create_model
import numpy as np
import h5py
import tensorflow as tf
import code
import tifffile

f = h5py.File("sample_A_20160501.hdf", "r")
raw_data=f["volumes"]["raw"].value

affinities=np.load("affinities.npy")

raw_data=np.asarray(raw_data)




#grab slice from the middle to test repetitively to see how loss decreases
raw_testing_data=np.zeros((1,1,16,128,128))
affinities_testing_data=np.zeros((1,3,16,128,128))
raw_testing_data[0]=raw_data[100:116,500:628,500:628]
affinities_testing_data[0]=affinities[:,100:116,500:628,500:628]




iteration=20000



init, train_op, raw_input, target, loss, out, c2 =create_model.create_model((None,1,16,128, 128),(None,3,16, 128, 128))

#from tensorflow.python.tools import inspect_checkpoint as chkp

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "saved/model%i"% iteration)
    print("Predicting...")
    out= sess.run(out, feed_dict={raw_input: raw_testing_data, target: affinities_testing_data})


tifffile.imsave("tiffs/predicted_affins",np.asarray(out[0,1,0,:,:],dtype=np.float32))
tifffile.imsave("tiffs/actual_affins",np.asarray(affinities_testing_data[0,1,0,:,:],dtype=np.float32))
tifffile.imsave("tiffs/raw",np.asarray(raw_testing_data[0,0,0,:,:],dtype=np.float32))