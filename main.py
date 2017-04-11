import numpy as np
import random
from ops import *
from model import *
import tensorflow as tf
import dataset

tf.set_random_seed(123)
np.random.seed(123)
random.seed(123)

TARGET = 'horse2zebra'

LOG_DIR = './log/'+TARGET

A_DIR = './datasets/'+TARGET+'/trainA/*'
B_DIR = './datasets/'+TARGET+'/trainB/*'

LEARNING_RATE = 0.0001
BETA_1 = 0.5
BETA_2 = 0.9

LAMBDA = 10
LAMBDA_CYCLE = 10

BATCH_SIZE = 8

MAX_ITERATION = 1000000
SAVE_PERIOD = 10000
SUMMARY_PERIOD = 50

NUM_CRITIC_TRAIN = 4

#############################################3
# Define Network
#############################################3
_, a = dataset.get_image_batch(A_DIR,BATCH_SIZE,300,256)
_, b = dataset.get_image_batch(B_DIR,BATCH_SIZE,300,256)

with tf.variable_scope('gen_a_to_b') as a_to_b_scope :
    b_gen = build_enc_dec(a)
with tf.variable_scope('gen_b_to_a') as b_to_a_scope :
    a_gen = build_enc_dec(b)

with tf.variable_scope('gen_b_to_a',reuse=True) :
    a_identity = build_enc_dec(b_gen,True)
with tf.variable_scope('gen_a_to_b',reuse=True) :
    b_identity = build_enc_dec(a_gen,True)

with tf.variable_scope('c_a') as scope:
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    a_hat = alpha * a+ (1.0-alpha) * a_gen

    v_a_real = build_critic(a)
    scope.reuse_variables()
    v_a_gen  = build_critic(a_gen)
    v_a_hat  = build_critic(a_hat)
with tf.variable_scope('c_b') as scope:
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    b_hat = alpha * b+ (1.0-alpha) * b_gen

    v_b_real = build_critic(b)
    scope.reuse_variables()
    v_b_gen  = build_critic(b_gen)
    v_b_hat  = build_critic(b_hat)

c_vars = [v for v in tf.trainable_variables() if v.name.startswith('c_')]
g_vars = [v for v in tf.trainable_variables() if v.name.startswith('gen_')]

#for v in c_vars : print v
#print('----------------------')
#for v in g_vars : print v

##################################
# Define Loss
##################################

c_optimizer = tf.train.AdamOptimizer(LEARNING_RATE,BETA_1,BETA_2)
g_optimizer = tf.train.AdamOptimizer(LEARNING_RATE,BETA_1,BETA_2)

# Training ops
W_a = tf.reduce_mean(v_a_real) - tf.reduce_mean(v_a_gen)
W_b = tf.reduce_mean(v_b_real) - tf.reduce_mean(v_b_gen)
W = W_a + W_b

GP_a = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.gradients(v_a_hat,a_hat)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
     )
GP_b = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.gradients(v_b_hat,b_hat)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
     )
GP = GP_a + GP_b

loss_c = -1.0*W + LAMBDA*GP
with tf.variable_scope('c_train') :
    gvs = c_optimizer.compute_gradients(loss_c,var_list=c_vars)
    train_c_op = c_optimizer.apply_gradients(gvs)

loss_g_a = -1.0 * tf.reduce_mean(v_a_gen)
loss_g_b = -1.0 * tf.reduce_mean(v_b_gen)
loss_g = loss_g_a + loss_g_b

loss_cycle_a = tf.reduce_mean(
    tf.reduce_mean(tf.abs(a - a_identity),reduction_indices=[1,2,3])) # following the paper implementation.(divide by #pixels)
loss_cycle_b = tf.reduce_mean(
    tf.reduce_mean(tf.abs(b - b_identity),reduction_indices=[1,2,3])) # following the paper implementation.(divide by #pixels)
loss_cycle = loss_cycle_a + loss_cycle_b

with tf.variable_scope('g_train') :
    gvs = g_optimizer.compute_gradients(loss_g+LAMBDA_CYCLE*loss_cycle,var_list=g_vars)
    train_g_op  = g_optimizer.apply_gradients(gvs)

#################################
# Miscellaneous(summary, init, etc.)
#################################
tf.summary.image('real_a',tf.transpose(a,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('fake_a',tf.transpose(a_gen,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('identity_a',tf.transpose(a_identity,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('real_b',tf.transpose(b,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('fake_b',tf.transpose(b_gen,perm=[0,2,3,1]),max_outputs=10)
tf.summary.image('identity_b',tf.transpose(b_identity,perm=[0,2,3,1]),max_outputs=10)

tf.summary.scalar('Estimated W',W)
tf.summary.scalar('gradient_penalty',GP)
tf.summary.scalar('loss_g', loss_g)
tf.summary.scalar('loss_cycle', loss_cycle)

# Summary Operations
summary_op = tf.summary.merge_all()

# Init operation
init_op = tf.global_variables_initializer()

#################################
# Train! (summary, init, etc.)
#################################

# Saver & Summary Writer
saver = tf.train.Saver(max_to_keep = 5)

# Queue ,Threads and Summary Writer
sess = tf.Session()
sess.run([init_op])

# if model exist, restore
"""
#if model exist :
#    saver.restore(sess,"path_to_model")
"""

try:
    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for step in xrange(MAX_ITERATION+1) :
        if coord.should_stop() :
            break

        for _ in xrange(NUM_CRITIC_TRAIN) :
            _ = sess.run(train_c_op)
        W_eval, GP_eval, loss_g_eval, loss_cycle_eval, _ = sess.run([W,GP,loss_g,loss_cycle,train_g_op])

        print('%7d : W : %1.6f, GP : %1.6f, Loss G : %1.6f, Loss Cycle : %1.6f'%(
            step,W_eval,GP_eval,loss_g_eval,loss_cycle_eval))
        if( step % SUMMARY_PERIOD == 0 ) :
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str,step)
        if( step % SAVE_PERIOD == 0 ):
            saver.save(sess,LOG_DIR+'/model.ckpt',global_step=step)

except Exception, e:
    coord.request_stop(e)
finally :
    coord.request_stop()
    coord.join(threads)

    sess.close()
