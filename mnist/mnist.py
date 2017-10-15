
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


# In[ ]:

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# In[ ]:

def inference(images, hidden_units1, hidden_units2):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden_units1], 
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name = 'weights')
        biases = tf.Variable(tf.zeros([hidden_units1]),
                            name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    
    # Hidden 1
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_units1, hidden_units2], 
                                stddev=1.0 / math.sqrt(float(hidden_units1))),
            name = 'weights')
        biases = tf.Variable(tf.zeros([hidden_units2]),
                            name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
    with tf.name_scope('softmax2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_units2, NUM_CLASSES], 
                                stddev=1.0 / math.sqrt(float(hidden_units2))),
            name = 'weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                            name = 'biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy)

def training(loss, training_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(training_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


# In[ ]:



