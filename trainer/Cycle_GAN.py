
# coding: utf-8

# In[1]:

import tensorflow as tf
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.lib.io import file_io
import argparse
from PIL import Image

# In[2]:

def instance_norm(x, reuse=False):
    with tf.variable_scope('instance_norm', reuse=reuse):
        out=tf.contrib.layers.instance_norm(x)
        return out


# In[3]:

def res_block(conv_data, filter_s, name='resnet'):
    with tf.variable_scope(name):
        x=conv_data
        conv_data=tf.pad(conv_data, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        gz1=tf.layers.conv2d(inputs=conv_data,filters=filter_s, kernel_size=[3,3], strides=(1,1), padding='valid' , name='rr1')
        ga1= tf.nn.relu(tf.contrib.layers.instance_norm(gz1))
        ga1 = tf.pad(ga1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        gz2=tf.layers.conv2d(inputs=ga1,filters=filter_s, kernel_size=[3,3], strides=(1,1), padding='valid', name='rr2')
        
        return tf.nn.relu(instance_norm(gz2)+instance_norm(x,reuse=True))


# In[4]:

#we find features of A convert them to features of B then deconvolve to B.
def generator(image_A, name='generator', reuse=False):       #image_A is 256*256*3
    with tf.variable_scope(name, reuse=reuse):
        image_A = tf.pad(image_A,[[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        gz1=tf.layers.conv2d(inputs=image_A,filters=32, kernel_size=[7,7], strides=(1,1), padding='valid', name='c1')
        ga1= tf.nn.relu(tf.contrib.layers.instance_norm(gz1))
        gz2=tf.layers.conv2d(inputs=ga1,filters=64, kernel_size=[3,3], strides=(2,2), padding='same', name='c2')
        ga2=tf.nn.relu(tf.contrib.layers.instance_norm(gz2))
        gz3=tf.layers.conv2d(inputs=ga2,filters=128, kernel_size=[3,3], strides=(2,2), padding='same', name='c3')
        
        #now we have 64*64*256 image, we send it to residual block for transformation of features
        res1=res_block(gz3,128, name='r1')
        res2=res_block(res1,128, name='r2')
        res3=res_block(res2,128, name='r3')
        res4=res_block(res3,128, name='r4')
        res5=res_block(res4,128, name='r5')
        res6=res_block(res5,128, name='r6')
        
        #deconvolve the features,
        gz4=tf.layers.conv2d_transpose(inputs=res6,filters=64, kernel_size=[3,3], strides=(2,2), padding='same', name='c4')
        ga4= tf.nn.relu(tf.contrib.layers.instance_norm(gz4))
        gz5=tf.layers.conv2d_transpose(inputs=ga4,filters=32, kernel_size=[3,3], strides=(2,2), padding='same', name='c5')
        ga5=tf.nn.relu(tf.contrib.layers.instance_norm(gz5))
        ga5 = tf.pad(ga5,[[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        gz6=tf.layers.conv2d(inputs=ga5,filters=3, kernel_size=[7,7], strides=(1,1), padding='valid', name='c6')
        output=tf.nn.tanh(gz6)
        return output


# In[5]:

def discriminator(image_B, name='discriminator', reuse=False):
    #x is of size 256*256*3
    with tf.variable_scope(name, reuse=reuse):
        
        dz1=tf.layers.conv2d(inputs=image_B,filters=64, kernel_size=[4,4], strides=(2,2), padding='same', name='c1')
        da1= tf.nn.leaky_relu(dz1,.2)
        dz2=tf.layers.conv2d(inputs=da1,filters=128, kernel_size=[4,4], strides=(2,2), padding='same', name='c2')
        da2= tf.nn.leaky_relu(tf.contrib.layers.instance_norm(dz2),.2)
        dz3=tf.layers.conv2d(inputs=da2,filters=256, kernel_size=[4,4], strides=(2,2), padding='same', name='c3')
        da3= tf.nn.leaky_relu(tf.contrib.layers.instance_norm(dz3),.2)
        dz4=tf.layers.conv2d(inputs=da3,filters=512, kernel_size=[4,4], strides=(1,1), padding='same', name='c4')
        da4= tf.nn.leaky_relu(tf.contrib.layers.instance_norm(dz4),.2)
        dz5=tf.layers.conv2d(inputs=da4,filters=1, kernel_size=[4,4], strides=(1,1), padding='same', name='c5')
        
        output=dz5
        return output


# In[6]:

def load_data(path):
    images=[]
    for filename in  file_io.get_matching_files(path+'/*.jpg'): 
        with file_io.FileIO(filename, mode='rb') as input_f :
            with file_io.FileIO( '1.jpg' , mode='w+') as output_f:
                            output_f.write(input_f.read())
        im=Image.open('1.jpg')
        im=np.asarray(im)
        if len(im.shape)==3:
            images.append(im)
    return images


# In[7]:

def fake_pool(count,img, fake_images, pool_size=50):
    if count<50:
        fake_images.append(img)
        return img
    else:
        p = np.random.random()
        if p > 0.5:
            random_id = np.random.randint(0,pool_size)
            temp = fake_images[random_id]
            fake_images[random_id] = img
            return temp
        else :
            return img


# In[8]:
def main(job_dir,**args):
    batch_size = 1
    lr = 0.0002
    epochs = 100
    img_h=256
    img_w=256

    train_data_A_path=job_dir+'train_horse'
    train_data_B_path=job_dir+'train_zebra'

    train_A_images=load_data(train_data_A_path)
    train_B_images=load_data(train_data_B_path)

    data_len=min(len(train_A_images),len(train_B_images))

        
    image_A = tf.placeholder(tf.float32, shape=(batch_size, img_h, img_w, 3))
    image_B = tf.placeholder(tf.float32, shape=(batch_size, img_h, img_w, 3))

    genB=generator(image_A, name='g_A')
    genA=generator(image_B, name='g_B')
    disA=discriminator(image_A, name='d_A')
    disB=discriminator(image_B, name='d_B')
    disgenA=discriminator(genA, name= 'd_A', reuse=True)
    disgenB=discriminator(genB, name='d_B', reuse=True)
    cycA=generator(genB, name='g_B', reuse=True)
    cycB=generator(genA, name='g_A', reuse=True)

    DAloss1=tf.reduce_mean(tf.squared_difference(disA,1))

    DBloss1=tf.reduce_mean(tf.squared_difference(disB,1))

    DAloss2=tf.reduce_mean(tf.square(disgenA))

    DBloss2=tf.reduce_mean(tf.square(disgenB))

    DAloss=(DAloss1+DAloss2)/2.0
    DBloss=(DBloss1+DBloss2)/2.0

    GBloss1 = tf.reduce_mean(tf.squared_difference(disgenA,1))   #the generator that generates A i.e.g_B, how bad is it
    GAloss1 = tf.reduce_mean(tf.squared_difference(disgenB,1))

    cycloss=tf.reduce_mean(tf.abs(image_A-cycA)) + tf.reduce_mean(tf.abs(image_B-cycB))

    GAloss=GAloss1+10*cycloss
    GBloss=GBloss1+10*cycloss

    #trainable params
    T_vars = tf.trainable_variables()
    D_Avars = [var for var in T_vars if 'd_A' in var.name]
    G_Avars = [var for var in T_vars if 'g_A' in var.name]
    D_Bvars = [var for var in T_vars if 'd_B' in var.name]
    G_Bvars = [var for var in T_vars if 'g_B' in var.name]

    #we get all vars and then update all functions like relu before calculating loss after each training
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_Aoptim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(DAloss, var_list=D_Avars)
        G_Aoptim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(GAloss, var_list=G_Avars)
        D_Boptim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(DBloss, var_list=D_Bvars)
        G_Boptim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(GBloss, var_list=G_Bvars)

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    tf.global_variables_initializer().run()
    logs_path=job_dir+'summary'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    GAloss1 = tf.placeholder(tf.float32)
    GBloss1 = tf.placeholder(tf.float32)
    DAloss1 = tf.placeholder(tf.float32)
    DBloss1 = tf.placeholder(tf.float32)
    tf.summary.scalar("galoss", GAloss1)
    tf.summary.scalar("gbloss", GBloss1)
    tf.summary.scalar("daloss", DAloss1)
    tf.summary.scalar("dbloss", DBloss1)
    summ=tf.summary.merge_all()
       # train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
       # train_set = (train_set - 0.5) / 0.5  #-1 to 1 normalize

    root =job_dir+ 'CycleGAN_results/'
    model = 'CycleGAN'
    saver = tf.train.Saver()




    # In[9]:


    with tf.device('/device:GPU:0'):
        for epoch in range(epochs):
            fake_images=0
            fake_A_images=[]
            fake_B_images=[]
            GAlosses=[]
            GBlosses=[]
            DAlosses=[]
            DBlosses=[]
            for iter in range(np.int(data_len/4)):
                imageA_ = train_A_images[iter*batch_size:(iter+1)*batch_size]
                imageB_ = train_B_images[iter*batch_size:(iter+1)*batch_size]
                #G_A network
                loss_g_A, _, fakeB = sess.run([GAloss, G_Aoptim, genB], {image_A: imageA_, image_B: imageB_})
                fakeB_temp= fake_pool(fake_images, fakeB, fake_B_images)
                GAlosses.append(loss_g_A)
                # D_B network
                loss_d_B, _ = sess.run([DBloss, D_Boptim], {image_A: imageA_, image_B: fakeB_temp})
                DBlosses.append(loss_d_B)
                #G_B network
                loss_g_B, _, fakeA = sess.run([GBloss, G_Boptim, genA], {image_A: imageA_, image_B: imageB_})
                fakeA_temp= fake_pool(fake_images, fakeA, fake_A_images)
                GBlosses.append(loss_g_B)
                #D_A network
                loss_d_A, _ = sess.run([DBloss, D_Boptim], {image_A: imageA_, image_B: fakeA_temp})
                DAlosses.append(loss_d_A)
                fake_images+=1

            s = sess.run(summ, feed_dict={GAloss1:np.mean(GAlosses), GBloss1:np.mean(GBlosses),DAloss1:np.mean(DAlosses), DBloss1:np.mean(DBlosses)})
            summary_writer.add_summary(s, epoch)
            summary_writer.flush()


            if epoch%9==0:
                saver.save(sess, './modelCycleGANepoch'+str(epoch)+'.ckpt')
                with file_io.FileIO('./modelCycleGANepoch'+str(epoch)+'.ckpt.data-00000-of-00001', mode='rb') as input_f :
                    with file_io.FileIO(job_dir+'modelCycleGANepoch'+str(epoch)+'.ckpt.data-00000-of-00001', mode='w+') as output_f:
                        output_f.write(input_f.read())
                with file_io.FileIO('./modelCycleGANepoch'+str(epoch)+'.ckpt.index', mode='rb') as input_f :
                    with file_io.FileIO(job_dir+'modelCycleGANepoch'+str(epoch)+'.ckpt.index', mode='w+') as output_f:
                        output_f.write(input_f.read())
                with file_io.FileIO('./modelCycleGANepoch'+str(epoch)+'.ckpt.meta', mode='rb') as input_f :
                    with file_io.FileIO(job_dir+'modelCycleGANepoch'+str(epoch)+'.ckpt.meta', mode='w+') as output_f:
                        output_f.write(input_f.read())  

            if epoch%4==0:
                for i in range(10):
                        fake_a,fake_b=sess.run([genA,genB],  {image_A: train_A_images[i:i+1], image_B: train_B_images[i:i+1]})
                        fake_a=tf.unstack(fake_a,axis=0)
                        fake_a=fake_a[0].eval()
                        fake_b=tf.unstack(fake_b,axis=0)
                        fake_b=fake_b[0].eval()

                        #combining pics
                        # im1=np.concatenate((train_A_images[i],fake_b),axis=2)
                        # im2=np.concatenate((train_B_images[i],fake_a),axis=2)
                        # fake_b=np.squeeze(im1, axis=0)
                        # fake_a=np.squeeze(im2, axis=0)


                        fixed_p1 = './' + model + 'fake_A' + str(i) + '.jpg'
                        fixed_p2 = './'+  model + 'fake_B' + str(i) + '.jpg'
                        plt.imsave(fixed_p1, fake_a)
                        plt.imsave(fixed_p2, fake_b)
                        with file_io.FileIO(fixed_p1, mode='rb') as input_f :
                                with file_io.FileIO(job_dir+'result/epoch'+str(epoch)+'/'+'fake_a'+str(i)+ '.jpg' , mode='w+') as output_f:
                                    output_f.write(input_f.read())
                        with file_io.FileIO(fixed_p2, mode='rb') as input_f :
                                with file_io.FileIO(job_dir+'result/epoch'+str(epoch)+'/'+'fake_b'+str(i)+ '.jpg' , mode='w+') as output_f:
                                    output_f.write(input_f.read())


    sess.close()  
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)




