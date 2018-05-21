import tensorflow as tf

def create_model():

    raw_in_sh = (1, 128, 128, 16)
    output_sh = (3, 128, 128, 16)

    raw_input= tf.placeholder(tf.float32,shape=raw_in_sh)
    target_input=tf.placeholder(tf.int16,shape=output_sh)

    
