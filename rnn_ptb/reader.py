
# coding: utf-8

# In[ ]:

'''Helper functions for parsing Penn TreeBank dataset'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections
import os
import sys
import tensorflow as tf
Py3 = sys.version_info[0] == 3


# In[ ]:

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# In[ ]:

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


# In[ ]:

a = tf.convert_to_tensor([1,2,3,4])
tf.size(a)
raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
len(raw_data)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#threads = tf.train.start_queue_runners(sess)
#i = tf.train.range_input_producer(limit=2, shuffle=False).dequeue()
#print (sess.run([i]))
#coord.request_stop()
#sess.run(model.queue.close(cancel_pending_enqueues=True))
#coord.join(threads)


# In[ ]:

def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size,
                                       message="epoch_size == 0")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size)
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], 
                            [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                            [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x,y


# In[ ]:

# !jupyter nbconvert --to script reader.ipynb


# In[ ]:



