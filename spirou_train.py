import create_model2 as create_model
import numpy as np
import h5py
import tensorflow as tf
import random_provider as rp
import code
import tifffile


raw_data=np.load("spir_raw.npy")
affinities=np.load("spir_aff.npy")

raw_data=np.asarray(raw_data)


#grab slice from the middle to test repetitively to see how loss decreases
raw_testing_data=np.zeros((1,1,16,128,128))
affinities_testing_data=np.zeros((1,3,16,128,128))
raw_testing_data[0,0]=raw_data[200:216,200:328,200:328]
affinities_testing_data[0]=affinities[:,200:216,200:328,200:328]





init, train_op, raw_input, target, loss,out, c2 =create_model.create_model((None,1,16,128, 128),(None,3,16, 128, 128))


with tf.Session() as sess:
    tf.summary.scalar("loss", loss)
    train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
    sess.run(init)



    n=0
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
        loss_sum, a= sess.run([loss,train_op],feed_dict={raw_input:raw_in,target:affin_in})


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
        #########################################
        # Print Visual Output for verification  #
        #########################################

        
        if (i % 10 == 0):

            print("Getting Output...")
            act_affins,pred = sess.run([target,out], feed_dict={raw_input: raw_testing_data, target: affinities_testing_data})

            # test for bad convergence
            if (n > 2):
                if (np.array_equal(pred_old, pred)):
                    print("CONVERGED TO BAD RESULT")
                    break

            tifffile.imsave("tiffs_spirou/prog/predicted_affins%i" % i,
                            np.asarray(pred, dtype=np.float32)[0, 1, 12, :, :])

            for j in range(0, 16):
                tifffile.imsave("tiffs_spirou/pred/predicted_affins%i" % j,
                                np.asarray(pred, dtype=np.float32)[0, 1, j, :, :])
                tifffile.imsave("tiffs_spirou/act/actual_affins%i" % j,
                                np.asarray(act_affins, dtype=np.float32)[0, 1, j, :, :])
                tifffile.imsave("tiffs_spirou/raw/raw%i" % j,
                                np.asarray(raw_testing_data[0, 0, j, :, :], dtype=np.float32))

            print("Saved Output, Raw and Affins as Tiffs")
            pred_old = pred


        ###########################################
        # Save Based on Iterations to Save Memory #
        ###########################################

        if (i < 100):
            if (i % 10 == 0):
                saver.save(sess, "./saved_spirou/model{}".format(i))
        if(i>=100 and i<1000):
            if (i % 100 == 0):
                saver.save(sess, "./saved_spirou/model{}".format(i))
        if(i>=1000 and i<1000):
            if(i%1000==0):
                saver.save(sess,"./saved_spirou/model{}".format(i))
        if(i>=10000 and i<100000):
            if (i % 5000 == 0):
                saver.save(sess, "./saved_spirou/model{}".format(i))
        if(i>=100000):
            if (i % 20000 == 0):
                saver.save(sess, "./saved_spirou/model{}".format(i))
        n+=1



print("DONE")

