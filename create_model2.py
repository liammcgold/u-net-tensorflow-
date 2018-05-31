import tensorflow as tf






def create_model(raw_in_sh,output_sh):


    #### input
    raw_input= tf.placeholder(tf.float32,shape=raw_in_sh,name="raw_input")
    input=tf.transpose(raw_input,(0,2,3,4,1))
    target=tf.placeholder(tf.float32,shape=output_sh,name="affins")

    #### NODE c0
    c0=tf.layers.conv3d(input,
                        8,
                        kernel_size=(16,16,16),
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )

    #### Node d0
    d0=tf.layers.max_pooling3d(c0,
                               pool_size=(16,64,64),
                               strides=(1,2,2),
                               padding="same"
                                )

    #### Node c1
    c1=tf.layers.conv3d(d0,
                        32,
                        kernel_size=(16,16,16),
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )

    #### Node d1
    d1=tf.layers.max_pooling3d(c1,
                               pool_size=(16,32,32),
                               strides=(1,2,2),
                               padding="same"
                                )

    #### Node c2
    c2=tf.layers.conv3d(d1,
                        64,
                        kernel_size=(16,16,16),
                        strides=[1,1, 1],
                        data_format="channels_last",
                        padding='SAME',
                        activation=tf.nn.relu
                        )

    #### Node d2
    d2=tf.layers.max_pooling3d(c2,
                               pool_size=(16,16,16),
                               strides=(1,2,2),
                               padding="same"
                                )

    #### Node c3
    c3=tf.layers.conv3d(d2,
                        64,
                        kernel_size=(16,16,16),
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
                         kernel_size=(16, 16, 16),
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
                           kernel_size=(16, 16, 16),
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
                           kernel_size=(16, 16, 16),
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


    #### OPTIMIZER
    optimizer=tf.train.AdamOptimizer(learning_rate=.00025)


    train_op=optimizer.minimize(loss)

    #### INIT
    init=tf.global_variables_initializer()
    return init, train_op, raw_input, target, loss, out, c2


