import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  embed_size = 50
  batch_size = 84
  label_size = 5
  hidden_size = 100
  max_epochs = 24
  early_stopping = 2
  dropout = 0.97
  lr = 0.001
  l2 = 0.0003
  window_size = 3

class NERModel(LanguageModel):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.
  """

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    self.wv, word_to_num, num_to_word = ner.load_wv(
      'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
    tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    self.num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in self.num_to_tag.iteritems()}

    # Load the training set
    docs = du.load_dataset('data/ner/train')
    self.X_train, self.y_train = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    self.X_dev, self.y_dev = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    docs = du.load_dataset('data/ner/test.masked')
    self.X_test, self.y_test = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(tf.int32, shape = [None, self.config.window_size])
    self.labels_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.label_size])
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE
    if label_batch is None:
      feed_dict = {self.input_placeholder: input_batch}
    else:
      feed_dict = {self.input_placeholder:input_batch, self.labels_placeholder:label_batch}
    ### END YOUR CODE
    return feed_dict

  def add_embedding(self):
    """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    (Words are discrete entities. They need to be transformed into vectors for use
    in deep-learning. Although we won't do so in this problem, in practice it's
    useful to initialize the embedding with pre-trained word-vectors. For this
    problem, using the default initializer is sufficient.)

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: See following link to understand what -1 in a shape means.
      https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.wv), embed_size)

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      embeddings = tf.get_variable("embeddings", [len(self.wv), self.config.embed_size],initializer=xavier_weight_init())
      embed = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      window = tf.reshape(embed, shape=[-1, self.config.window_size*self.config.embed_size])

      ### END YOUR CODE
      return window

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    ### YOUR CODE HERE
    '''
    declare W and b1 in variable scope and add regularization in get_variable() so that in loss function you can directly use tf.get_collection to retrieve L2 regularization.
    if W and b1 are declared outside the variable scope, tf.add_to_collection is used to manually add regularization to the collection and then retrive in loss function.
    '''

    with tf.variable_scope("Layer"):
      W = tf.get_variable("W", [self.config.window_size * self.config.embed_size, self.config.hidden_size],
                            dtype=tf.float32, initializer=xavier_weight_init(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.l2))
      b1 = tf.get_variable("b1", [self.config.hidden_size, ])
      h = tf.nn.relu(tf.matmul(window, W) + b1)
      h = tf.nn.dropout(h, self.config.dropout)

    with tf.variable_scope("Softmax"):
      U = tf.get_variable("U", [self.config.hidden_size, self.config.label_size], dtype=tf.float32,
                            initializer=xavier_weight_init(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.l2))
      b2 = tf.get_variable("b2", [self.config.label_size])

      output = tf.matmul(h,U) + b2
      output = tf.nn.dropout(output,self.config.dropout)

    ### END YOUR CODE
    return output 

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    '''''
    use tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) to retrieve L2 regularization
    tf.nn.softmax_cross_entropy_with_logits performs softmax and cross_entropy
    '''''
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = self.labels_placeholder)
    loss = tf.reduce_mean(loss)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    '''
    optimizer needs specified learning rate
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op = optimizer.minimize(loss)
    ### END YOUR CODE
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    window = self.add_embedding()
    y = self.add_model(window)

    self.loss = self.add_loss_op(y)

    self.trainloss = tf.summary.scalar('trainloss', self.loss)
    self.validloss = tf.summary.scalar('validloss', self.loss)

    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)

  def run_epoch(self, session, input_data, input_labels, epoch_number, result_writer,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _= session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)

      summary = session.run(self.trainloss, feed_dict=feed)
      result_writer.add_summary(summary, epoch_number*total_steps+step)

      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()

    train_loss = np.mean(total_loss)
    train_accuracy = total_correct_examples / float(total_processed_examples)


    return train_loss, train_accuracy

  def predict(self, session, X, y, result_writer):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        summary, loss, preds = session.run(
            [self.validloss, self.loss, self.predictions], feed_dict=feed)
        if result_writer is not None:
            result_writer.add_summary(summary, step)

        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)

    validation_loss = np.mean(losses)

    return validation_loss, results

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print ()
    print (confusion)
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print ('Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall))

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in range(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write(str(prediction) + "\n")

def test_NER():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = NERModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0
      result_writer = tf.summary.FileWriter('./result_log', session.graph)
      session.run(init)

      for epoch in range(config.max_epochs):
        print ('Epoch {}'.format(epoch))
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train, epoch, result_writer)
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev, result_writer)

        print ('Training loss: {}'.format(train_loss))
        print ('Training acc: {}'.format(train_acc))
        print ('Validation loss: {}'.format(val_loss))

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")
        
          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
        confusion = calculate_confusion(config, predictions, model.y_dev)
        print_confusion(confusion, model.num_to_tag)
        print ('Total time: {}'.format(time.time() - start))
      
      saver.restore(session, './weights/ner.weights')
      print ('Test')
      print ('=-=-=')
      print ('Writing predictions to q2_test.predicted')
      _, predictions = model.predict(session, model.X_test, model.y_test, None)
      save_predictions(predictions, "q2_test.predicted")

if __name__ == "__main__":
  test_NER()
