import numpy as np
import random
from ops import *
from model import *
import tensorflow as tf
import dataset

import sys
import os
from PIL import Image

tf.set_random_seed(123)
np.random.seed(123)
random.seed(123)

##########################################
# User defined variables
##########################################
TARGET = 'horse2zebra'
MODEL_FILE = './models/horse2zebra/model.ckpt-20000'


BATCH_SIZE = 1
A_DIR = './datasets/'+TARGET+'/testA/*'
B_DIR = './datasets/'+TARGET+'/testB/*'
RESULT_A_DIR = './results/'+TARGET+'/resultA/'
RESULT_B_DIR = './results/'+TARGET+'/resultB/'
if not os.path.exists(RESULT_A_DIR): os.makedirs(RESULT_A_DIR)
if not os.path.exists(RESULT_B_DIR): os.makedirs(RESULT_B_DIR)

#############################################3
# Define Network
#############################################3
f_a,a = dataset.get_image_batch(A_DIR,BATCH_SIZE,300,256,train=False)
f_b,b = dataset.get_image_batch(B_DIR,BATCH_SIZE,300,256,train=False)

with tf.variable_scope('gen_a_to_b') as a_to_b_scope :
    b_gen = build_enc_dec(a)
with tf.variable_scope('gen_b_to_a') as b_to_a_scope :
    a_gen = build_enc_dec(b)

with tf.variable_scope('gen_b_to_a',reuse=True) :
    a_identity = build_enc_dec(b_gen,True)
with tf.variable_scope('gen_a_to_b',reuse=True) :
    b_identity = build_enc_dec(a_gen,True)

#################################
# Miscellaneous(summary, init, etc.)
#################################
# Saver & Summary Writer
saver = tf.train.Saver(var_list=tf.trainable_variables())

# Queue ,Threads and Summary Writer
sess = tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

saver.restore(sess,MODEL_FILE)

def _convert_to_image(im) :
    x = np.transpose(im * 128 + 128,[0,2,3,1]) #TODO: or, [min to max]?
    x = np.squeeze(x).astype(np.uint8)
    return Image.fromarray(x)

def _save_to_file(dir,f,a,b,a_identity) :
    images = [_convert_to_image(i) for i in [a,b,a_identity]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    f = os.path.join(dir,os.path.basename(f))
    print(f)
    new_im.save(f)

try:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    while( not coord.should_stop() ):
        f_a_eval,a_eval,b_gen_eval,a_identity_eval = sess.run([f_a,a,b_gen,a_identity])
        _save_to_file(RESULT_A_DIR,f_a_eval[0],a_eval,b_gen_eval,a_identity_eval)

        f_b_eval,b_eval,a_gen_eval,b_identity_eval = sess.run([f_b,b,a_gen,b_identity])
        _save_to_file(RESULT_B_DIR,f_b_eval[0],b_eval,a_gen_eval,b_identity_eval)
except Exception, e:
    coord.request_stop(e)
finally :
    coord.request_stop()
    coord.join(threads)

    sess.close()
