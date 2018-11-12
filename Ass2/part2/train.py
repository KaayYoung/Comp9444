"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""

import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

# Added
import sys

import implementation as imp

batch_size = imp.batch_size
#iterations = 100000
iterations = 31000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

#def getTrainBatch():
#    labels = []
#    arr = np.zeros([batch_size, seq_length])
#    for i in range(batch_size):
#        if (i % 2 == 0):
#            num = randint(0, 12499)
#            labels.append([1, 0])
#        else:
#            num = randint(12500, 24999)
#            labels.append([0, 1])
#        arr[i] = training_data[num]
#    return arr, labels

# Remove next 2 functions
def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 11499)
            labels.append([1, 0])
        else:
            num = randint(13500, 24999)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = randint(11500, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
input_data, labels, optimizer, accuracy, loss = imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        writer.add_summary(summary, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)
sess.close()

lstm_size = 32
classes = 2
sentence_length = 40
word_vec_dim = 50
input_data = tf.placeholder(tf.int32, [batch_size, sentence_length])
labels = tf.placeholder(tf.float32, [batch_size, classes])

# Initialize states of LSTM to zero
# initial_state = tf.zeros([batch_size, lstm.state_size])
# state = tf.zeros([batch_size, lstm.state_size])

# Convert glove_embeddings_arr to tensor
tf_glove_embeddings_arr = tf.convert_to_tensor(glove_array)

# Put data in form suitable for feeding LSTM
# Get 3D tensor with word vectors for each word in a time-step
word_embeddings = tf.Variable(tf.zeros([batch_size, sentence_length,\
        word_vec_dim]),dtype=tf.float32)
word_embeddings = tf.nn.embedding_lookup(tf_glove_embeddings_arr,\
        input_data)

# Initialize LSTM cell
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

# Add dropout to reduce overfitting
#lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=\
        #        0.75)

# Unroll network dynamically
val, _ = tf.nn.dynamic_rnn(lstm, word_embeddings, dtype = \
        tf.float32)

# Declare weight and bias variables
weight = tf.Variable(tf.truncated_normal([lstm_size, classes]))
bias = tf.Variable(tf.constant(0.1, shape=[classes]))

# Takes hidden state and processes it for output
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
pred = (tf.matmul(last, weight) + bias)

curr_pred = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(curr_pred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

iterations = 100
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    
    print("Accuracy for this batch:", (sess.run(accuracy, \
            {input_data: nextBatch, labels: nextBatchLabels})) * 100)
sess.close()
