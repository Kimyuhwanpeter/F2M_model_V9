# -*- coding:utf-8 -*-
from F2M_model_V8_2 import *
from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 1024, 
                           
                           "load_size": 1044,

                           "tar_size": 1024,

                           "tar_load_size": 1044,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[3]detection_DB/CelebAMask-HQ/archive/celeba_hq/train_test/female_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[3]detection_DB/CelebAMask-HQ/archive/celeba_hq/train_test/female/",
                           
                           "B_txt_path": "D:/[1]DB/[5]4th_paper_DB/male_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AFAD/All/male_40_63/",

                           "age_range": [40, 64],

                           "n_classes": 24,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Pictures/img"})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data)
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data)
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_load_size, FLAGS.tar_load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    B_img = B_img / 127.5 - 1.

    return A_img, B_img

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def cal_loss(A2B_G_model, B2A_G_model, B_discriminator,
             A_batch_images, B_batch_images, extract_feature_model):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape() as d_tape:
        fake_B = model_out(A2B_G_model, [A_batch_images, B_batch_images], True)
        A_resize = tf.image.resize(A_batch_images, [FLAGS.tar_size, FLAGS.tar_size])
        fake_A_ = model_out(B2A_G_model, [fake_B, A_resize], True)

        # identification
        id_fake_A = model_out(B2A_G_model, [A_batch_images, B_batch_images], True)

        DB_real = model_out(B_discriminator, B_batch_images, True)
        fake_B_resize = tf.image.resize(fake_B, [FLAGS.tar_size, FLAGS.tar_size])
        DB_fake = model_out(B_discriminator, fake_B_resize, True)

        ################################################################################################
        # 나이에 대한 distance를 구하는곳
        # feature vector로 뽑아야하기 때문에 어떻게 뽑아야할지가 강권 (pre-train model을 사용할까?)
        vector_fake_B = model_out(extract_feature_model, fake_B, False)
        B_resize = tf.image.resize(B_batch_images, [FLAGS.img_size, FLAGS.img_size])
        vector_real_B = model_out(extract_feature_model, B_batch_images, False)
      
        realB_fakeB_en = tf.reduce_sum(tf.abs(vector_real_B - vector_fake_B), 1)
        realA_fakeB_loss = (2/100) * (realB_fakeB_en*realB_fakeB_en)

        # A와 B 나이가 다르면 감소함수, 같으면 증가함수

        loss_buf = 0.
        loss_buf2 = 0.
        for j in range(FLAGS.batch_size):
            loss_buf += realA_fakeB_loss[j]
        loss_buf /= FLAGS.batch_size
        ################################################################################################

        id_loss = tf.reduce_mean(tf.abs(id_fake_A - A_batch_images)) * 10.0

        Cycle_loss = (tf.reduce_mean(tf.abs(fake_A_ - A_batch_images))) * 10.0
        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + id_loss + loss_buf
        d_loss = Adver_loss

    g_grads = g_tape.gradient(g_loss, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables)
    d_grads = g_tape.gradient(d_loss, A_discriminator.trainable_variables + B_discriminator.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_discriminator.trainable_variables + B_discriminator.trainable_variables))

    return g_loss, d_loss

def main():
    extract_feature_model = tf.keras.applications.VGG16(include_top=False,
                                                        input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    h = extract_feature_model.output
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(FLAGS.n_classes)(h)
    extract_feature_model = tf.keras.Model(inputs=extract_feature_model.input, outputs=h)
    extract_feature_model.summary()

    A2B_G_model = F2M_generator_V2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), target_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))
    B2A_G_model = F2M_generator_V2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), target_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))
    B_discriminator = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    A2B_G_model.summary()
    B_discriminator.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):
            min_ = min(len(A_images), len(B_images))
            shuffle(A_images)
            shuffle(B_images)
            A_images = A_images[:min_]
            B_images = B_images[:min_]

            A_train_img, B_train_img = np.array(A_images), np.array(B_images)

            # 가까운 나이에 대해서 distance를 구하는 loss를 구성하면, 결국에는 해당이미지의 나이를 그대로 생성하는 효과?를 볼수있을것
            gener = tf.data.Dataset.from_tensor_slices((A_train_img, B_train_img))
            gener = gener.shuffle(len(B_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = len(A_train_img) // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, B_batch_images = next(train_it)

                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, B_discriminator,
                                          A_batch_images, B_batch_images, extract_feature_model)

                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))
                
                if count % 100 == 0:
                    fake_B = model_out(A2B_G_model, A_batch_images, False)
                    fake_A = model_out(B2A_G_model, B_batch_images, False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), fake_B[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/fake_A_{}.jpg".format(count), fake_A[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0] * 0.5 + 0.5)


                #if count % 1000 == 0:
                #    num_ = int(count // 1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} folder to store the weight!".format(num_))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                #                               A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                #                               g_optim=g_optim, d_optim=d_optim)
                #    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)

                count += 1


if __name__ == "__main__":
    main()
