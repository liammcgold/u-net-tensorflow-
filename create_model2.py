import tensorflow as tf





def create_model(raw_in_sh,output_sh):



#### input
    raw_input= tf.placeholder(tf.float32,shape=raw_in_sh,name="raw_input")
    input=tf.transpose(raw_input,(0,2,3,4,1))
    #input=tf.layers.batch_normalization(input)


    c0=tf.layers.conv3d(input,3,(16,16,16),padding="SAME",activation=tf.nn.relu)
    c0=tf.layers.conv3d(c0,3,(16,16,16),padding="SAME",activation=tf.nn.relu)
    out=tf.transpose(c0,(0,4,1,2,3))
    target=tf.placeholder(tf.float32,shape=output_sh)


    #### LOSS



    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=out)
    loss = tf.reduce_mean(loss)

    optimizer=tf.train.AdamOptimizer(learning_rate=.00005)
    train_op=optimizer.minimize(loss)




    #target=tf.placeholder(tf.float32,shape=output_sh,name="affins")

    #c0_w=tf.Variable(tf.random_normal([1,raw_input.shape[1].value,raw_input.shape[2].value,raw_input.shape[3].value,raw_input.shape[4].value]))

    #c0=tf.nn.conv3d(raw_input,c0_w,(1,1,1,2,2),padding="SAME",data_format="NCDHW")






#### INIT
    init=tf.global_variables_initializer()
    return init,raw_input,target,train_op,loss,out


#data goes in (b,f,z,y,x)
#model=create_model((None,1,16,128, 128),(None,3,16, 128, 128))
