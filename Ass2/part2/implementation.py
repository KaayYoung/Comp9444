import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile

# Added
import sys
import string
import re


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here.
    The first 12500 reviews in the array should be the positive reviews,
    the 2nd 12500 reviews should be the negative reviews. 
    RETURN: numpy array of data with each row being a review in
    vectorized form
    """
   
    table = str.maketrans("-.,/[]@$?{}<>&~()_%|*#;:!\"",\
            "                          ", "'`")
    rev_arr = np.array([], dtype=int)
    directories = ["pos", "neg"]
    for directory in directories:
        for file in glob.glob(r'{0}/*.txt'.format(directory)):
            review = open(file,'r',encoding="utf-8")
            review_txt = review.read()
            tags   = re.compile('<.*?>')
            review_txt = re.sub(tags, ' ', review_txt)
            review_txt = review_txt.translate(table).lower().split()
            count = 0
            tmp_arr = np.array([], dtype=int)
            for word in review_txt:
                if count >= 40:
                    break
                if word in glove_dict:
                    tmp_arr = np.append(tmp_arr, glove_dict[word])
                    count += 1
                else:
                    tmp_arr = np.append(tmp_arr, glove_dict['UNK'])
                    count += 1
            
            if tmp_arr.size < 40:
                for i in range(tmp_arr.size, 40):
                    tmp_arr = np.append(tmp_arr, 0)
            
            review.close()
            rev_arr = np.append(rev_arr, tmp_arr)
    
    """
    DEBUG
    #print(rev_arr)
    print("PreReShape: ", len(rev_arr))
    """
    
    data = rev_arr.reshape(25000, 40)
   
    """
    DEBUG
    print("AftReShape: ", len(rev_arr))
    print(len(rev_arr[0]))
    print(rev_arr[0])
    print(rev_arr[10])
    print(rev_arr[50])

    sys.exit()
    """
    
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into an array and a dictionary with words
    as keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named
    "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string
                             form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    # if you are running on the CSE machines, you can load the glove
    # data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",\
    # 'r',encoding="utf-8")
   
    # List with 50 zeros representing unknown words
    unk_array = [0] * 50
    embeddings = []
    #unk_array = np.array([0] * 50)
    #embeddings = np.array([])

    # Initialize embedding list and index dictionary
    embeddings.append(unk_array) 
    #embeddings = np.append(embeddings, unk_array) 
    word_index_dict = {'UNK': 0}

    for line in data:
        #line = line.rstrip()
        #(word, rest) = line.split(maxsplit=1)
        #rest_arr = np.fromstring(rest, dtype=float, sep=' ')
        #word_index_dict[word] = embeddings.size
        #embeddings = np.append(embeddings, rest_arr)
        line = line.rstrip()
        (word, rest) = line.split(maxsplit=1)
        rest_arr = rest.split()
        rest_arr = [float(number) for number in rest_arr]
        word_index_dict[word] = len(embeddings)
        embeddings.append(rest_arr)
    
    """
    Debug
    print("First line:  ", embeddings[0])
    print("First index: ", word_index_dict.get("UNK"))
    print("First line type:  ", type(embeddings[0]))
    
    print("Second line:  ", embeddings[1])
    print("Second index: ", word_index_dict.get("the"))
    print("Second line type:  ", type(embeddings[1]))
    sys.exit()
    """
    
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at
    least one recurrent unit. The input placeholder should be of size
    [batch_size, 40] as we are restricting each review to it's first 40
    words. The following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy
            and loss tensors
    """

    """
    words_in_dataset = tf.placeholder(tf.float32, [num_batches,\
            batch_size, num_features])
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Initial state of the LSTM memory.
    hidden_state  = tf.zeros([batch_size, lstm.state_size])
    current_state = tf.zeros([batch_size, lstm.state_size])
    state = hidden_state, current_state
    probabilities = []
    loss = 0.0
    for current_batch_of_words in words_in_dataset:
        # The value of state is updated after processing each batch of
        # words.
        output, state = lstm(current_batch_of_words, state)

        # The LSTM output can be used to make next word predictions
        logits = tf.matmul(output, softmax_w) + softmax_b
        probabilities.append(tf.nn.softmax(logits))
        loss += loss_function(probabilities, target_words)
    """
    # LSTM
    #   - lstm_size: Number of hidden units
    #   - num_classes: represents positive or negative
    #   - input_data: placeholder for sentences
    #   - labels: placeholder for positive/negative probability
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
    tf_glove_embeddings_arr = tf.convert_to_tensor(glove_embeddings_arr)
 
    # Put data in form suitable for feeding LSTM
    # Get 3D tensor with word vectors for each word in a time-step
    word_embeddings = tf.Variable(tf.zeros([batch_size, sentence_length,\
            word_vec_dim]),dtype=tf.float32)
    word_embeddings = tf.nn.embedding_lookup(tf_glove_embeddings_arr,\
            input_data)

    # Initialize LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to reduce overfitting
    lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=\
            0.75)

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

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
            logits=pred, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, optimizer, accuracy, loss
