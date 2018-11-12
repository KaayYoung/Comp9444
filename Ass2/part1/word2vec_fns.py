import tensorflow as tf
import numpy as np
import collections

data_index = 0

def generate_batch(data, batch_size, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds
        the training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right
        that form the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each
    entry for the batch containing all the context words, with the
    corresponding label being the word in the middle of the context
    """
    global data_index
    context_size = 2 * skip_window
    batch = np.ndarray(shape=(batch_size, context_size), \
            dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    target_word = span // 2 
   
    #Initialize the buffer and make sure index starts at 0
    if data_index + span > len(data):
        data_index = 0
    buffer = collections.deque(maxlen=span)
    buffer.extend(data[data_index:data_index + span])
    
    #Get indexes for context words
    context_words = [i for i in range(span) if i != span//2]

    for context in range(batch_size):
        current_span = np.array(buffer)
        
        # Extract context words from span and add them to the batch
        batch[context] = current_span[context_words]
   
        # Extract the label from the span
        labels[context, 0] = buffer[target_word]

        # Add a new word to the buffer and increment index for
        # next loop iteration
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    return batch, labels

def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings 
        (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs 
        (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all
        context words for each entry in the batch, should have shape
        (batch_size, embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        pass
    
    # Create embedding lookup (table?) then reduce the dimension
    # by getting average of columns (context words)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    return tf.reduce_mean(embed, 1)
    
