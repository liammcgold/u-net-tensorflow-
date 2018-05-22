import tensorflow as tf
import numpy as np


weights={}
biases={}




def create_model(raw_in_sh,output_sh):


#### Raw input
    raw_input= tf.placeholder(tf.float32,shape=raw_in_sh,name="raw_input")
    input=tf.reshape(raw_input,shape=(1,raw_in_sh[2],raw_in_sh[3],raw_in_sh[4],1))


#### NODE c0
    #Define variables for this layer
    filter_shape=(1,16,128,128,8)
    weights["c0"]=tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=.05))
    biases["c0"]=tf.Variable(tf.random_normal([8]))

    #create convolution
    c0=tf.layers.conv3d(input,                  #input node
                        8,                          #number of filters
                        kernel_size=(16,16,16),     #kernel size
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME'
                        )
    #create max pool
    d0=tf.layers.max_pooling3d(input,
                               pool_size=(16,64,32),
                               strides=(1,1,1),
                               padding="same"
                                )
    print(d0.shape)




    #need to reshape at end so that comparison to loss can be made!


    #c0=make_level_conv(raw_input,8,(1, 1, 16, 128, 128))
    #d0=make_level_pool(c0)





#data goes in (b,f,z,y,x)


create_model((None,1,16,128, 128),(None,3,16, 128, 128))