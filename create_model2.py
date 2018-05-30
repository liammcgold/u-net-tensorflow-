import tensorflow as tf






def create_model(raw_in_sh,output_sh):


#### input
    raw_input= tf.placeholder(tf.float32,shape=raw_in_sh,name="raw_input")
    input=tf.transpose(raw_input,(0,2,3,4,1))
    #input=tf.layers.batch_normalization(input)
    target=tf.placeholder(tf.float32,shape=output_sh,name="affins")



#### NODE c0
    #create convolution
    c0=tf.layers.conv3d(input,                      #input node
                        8,                          #number of filters
                        kernel_size=(16,16,16),     #kernel size
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )
#### Node d0
    #create max pool
    d0=tf.layers.max_pooling3d(c0,
                               pool_size=(16,64,64),
                               strides=(1,2,2),
                               padding="same"
                                )
#### Node c1

    #create convolution
    c1=tf.layers.conv3d(d0,                      #input node
                        32,                          #number of filters
                        kernel_size=(16,16,16),     #kernel size
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )
#### Node d1
    #create max pool
    d1=tf.layers.max_pooling3d(c1,
                               pool_size=(16,32,32),
                               strides=(1,2,2),
                               padding="same"
                                )



#### Node c2

    #create convolution
    c2=tf.layers.conv3d(d1,                      #input node
                        64,                          #number of filters
                        kernel_size=(16,16,16),     #kernel size
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )
#### Node d2
    #create max pool
    d2=tf.layers.max_pooling3d(c2,
                               pool_size=(16,16,16),
                               strides=(1,2,2),
                               padding="same"
                                )

#### Node c3

    #create convolution
    c3=tf.layers.conv3d(d2,                      #input node
                        64,                          #number of filters
                        kernel_size=(16,16,16),     #kernel size
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )

#### Node m0

    u0=tf.keras.layers.UpSampling3D((1,2,2))(d2)

    m0=tf.add(c2,u0)

#### Node mc0
    mc0=tf.layers.conv3d(m0,
                         32,
                         kernel_size=(16, 16, 16),  # kernel size
                         strides=[1, 1, 1],
                         data_format="channels_last",
                         padding='SAME',
                         activation=tf.nn.relu
                         )
#### Node m1
    u1=tf.keras.layers.UpSampling3D((1,2,2))(mc0)

    m1=tf.add(c1,u1)

#### Node mc1
    mc1 = tf.layers.conv3d(m1,
                           8,
                           kernel_size=(16, 16, 16),  # kernel size
                           strides=[1, 1, 1],
                           data_format="channels_last",
                           padding='SAME',
                           activation=tf.nn.relu
                           )
#### Node m2
    u2 = tf.keras.layers.UpSampling3D((1, 2, 2))(mc1)

    m2=tf.add(c0,u2)

#### Node mc2
    mc2 =  tf.layers.conv3d(m2,
                           3,
                           kernel_size=(16, 16, 16),  # kernel size
                           strides=[1, 1, 1],
                           data_format="channels_last",
                           padding='SAME',
                           activation=tf.nn.relu
                           )


#### Node out
    out=tf.transpose(mc2,(0,4,1,2,3))




#### LOSS
    loss=tf.nn.weighted_cross_entropy_with_logits(targets=target,logits=out, pos_weight=.05)
    loss=tf.reduce_mean(loss)

    #avrg=tf.reduce_mean(out)
    #out=tf.multiply(tf.divide(out,avrg),tf.constant(.5,dtype=tf.float32,shape=(1,3,16,128,128)))


    #out=tf.add(out,tf.constant(1,dtype=tf.float32,shape=(1,3,16,128,128)))
    #loss=tf.subtract(target,out)
    #loss=tf.square(loss)
    #loss=tf.reduce_mean(loss)

#### OPTIMIZER
    optimizer=tf.train.AdamOptimizer(learning_rate=.00025)


    train_op=optimizer.minimize(loss)

#### INIT
    init=tf.global_variables_initializer()
    return init, train_op, raw_input, target, loss, out, c2


#data goes in (b,f,z,y,x)
#model=create_model((None,1,16,128, 128),(None,3,16, 128, 128))
