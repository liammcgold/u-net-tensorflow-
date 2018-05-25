import create_model
import numpy as np
import h5py
import tensorflow as tf
import random_provider as rp
import code

f = h5py.File("sample_A_20160501.hdf", "r")
raw_data=f["volumes"]["raw"].value

affinities=np.load("affinities.npy")

raw_data=np.asarray(raw_data)


#grab slice from the middle to test repetitively to see how loss decreases
raw_testing_data=np.zeros((1,1,16,128,128))
affinities_testing_data=np.zeros((1,3,16,128,128))
raw_testing_data[0]=raw_data[100:116,500:628,500:628]
affinities_testing_data[0]=affinities[:,100:116,500:628,500:628]





init, train_op, raw_input, target, loss,out =create_model.create_model((None,1,16,128, 128),(None,3,16, 128, 128))


with tf.Session() as sess:
    tf.summary.scalar("loss", loss)
    train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
    sess.run(init)




    saver=tf.train.Saver()
    for i in range(1000000):
        print("Training itteration: ",i)

        #grab random slices
        raw_data_samp, affinities_samp=rp.random_provider((16,128,128),raw_data,affinities)
        raw_in=np.zeros((1,1,16,128,128))
        affin_in=np.zeros((1,3,16,128,128))
        raw_in[0]=raw_data_samp
        affin_in[0]=affinities_samp

        #run training
        merge = tf.summary.merge_all()
        summary, loss_sum, a= sess.run([merge,loss,train_op],feed_dict={raw_input:raw_in,target:affin_in})


        ###################################
        # Print Summary with testing data #
        ###################################

        if (i % 10 == 0):
            # perform retest of data
            print("TESTING....")
            loss_sum = sess.run([loss], feed_dict={raw_input: raw_testing_data, target: affinities_testing_data})[0]
            if(i>0):
                loss_dif_p=100*(loss_sum_old-loss_sum)/loss_sum_old
            print("TESTED LOSS = ",loss_sum)
            if(i>0):
                print("Percent Decrease =", int(loss_dif_p), "%")
            loss_sum_old = loss_sum


        ###########################################
        # Save Based on Iterations to Save Memory #
        ###########################################

        if (i < 100):
            if (i % 10 == 0):
                saver.save(sess, "./saved/model{}".format(i))
        if(i>=100 and i<1000):
            if (i % 100 == 0):
                saver.save(sess, "./saved/model{}".format(i))
        if(i>=1000 and i<1000):
            if(i%1000==0):
                saver.save(sess,"./saved/model{}".format(i))
        if(i>=10000 and i<100000):
            if (i % 5000 == 0):
                saver.save(sess, "./saved/model{}".format(i))
        if(i>=100000):
            if (i % 20000 == 0):
                saver.save(sess, "./saved/model{}".format(i))
