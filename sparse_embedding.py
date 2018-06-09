from __future__ import print_function

import collections
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics
import os
import urllib


#wget https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt -O terms.txt
tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_file = train_url.split("/")[-1]
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_file = test_url.split("/")[-1]
prefix ="/home/jjzhang/.keras/datasets/"
def download():
    if not os.path.exists(train_file):
        tf.keras.utils.get_file(train_file, train_url)
    if not os.path.exists(test_file):
        tf.keras.utils.get_file(test_file, test_url)

download()

def _parse_function(record):
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }
    parsed_features= tf.parse_single_example(record, features)
    terms = parsed_features["terms"].values
    labels = parsed_features["labels"]

    return {"terms": terms}, labels

'''
ds = tf.data.TFRecordDataset(prefix+train_file)
ds= ds.map(_parse_function)
next = ds.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    print(sess.run(next))
'''

def _input_fn(filenames, num_epochs=None, shuffle=True):
    ds = tf.data.TFRecordDataset(filenames=filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.padded_batch(25, ds.output_shapes)
    ds = ds.repeat(num_epochs)
    return ds.make_one_shot_iterator().get_next()
informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",
                                                                                 vocabulary_list=informative_terms)
optimizer = tf.train.AdagradOptimizer(
    learning_rate=0.1
)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

def linearTrain():

    classifier = tf.estimator.LinearClassifier(
        feature_columns=[terms_feature_column],
        optimizer=optimizer
    )
    classifier.train(input_fn=lambda : _input_fn([prefix+train_file]), steps=1000)
    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+train_file]),
                                             steps=1000)
    print("training set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+test_file]),
                                             steps=1000)
    print("test set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

#linearTrain()
def dnnTrain():

    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],
        optimizer=optimizer,
        hidden_units=[20, 20],
    )
    classifier.train(input_fn=lambda : _input_fn([prefix+train_file]), steps=1000)
    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+train_file]),
                                             steps=1000)
    print("training set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+test_file]),
                                             steps=1000)
    print("test set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

def dnnEmbedding():
    terms_embedding_coumn = tf.feature_column.embedding_column(
        categorical_column=terms_feature_column,
        dimension=2
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=[terms_embedding_coumn],
        optimizer=optimizer,
        hidden_units=[20, 20],
    )
    classifier.train(input_fn=lambda : _input_fn([prefix+train_file]), steps=1000)
    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+train_file]),
                                             steps=1000)

    print ("------------embedding------")
    print("training set metrics :")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+test_file]),
                                             steps=1000)
    print("test set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")
    return classifier
'''
print (classifier.get_variable_names())
print (classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape)

embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

for term_index in range(len(informative_terms)):
    # Create a one-hot encoding for our term. It has 0s everywhere, except for
    # a single 1 in the coordinate that corresponds to that term.
    term_vector = np.zeros(len(informative_terms))
    term_vector[term_index] = 1
    # We'll now project that one-hot vector into the embedding space.
    embedding_xy = np.matmul(term_vector, embedding_matrix)
    plt.text(embedding_xy[0],
             embedding_xy[1],
             informative_terms[term_index])

# Do a little setup to make sure the plot displays nicely.
plt.rcParams["figure.figsize"] = (15, 15)
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show()
'''


def dnnEmbeddingWithMoreTerms():
    category_coumn = tf.feature_column.categorical_column_with_vocabulary_file(
        key="terms",
        vocabulary_file="terms.txt"
    )
    terms_embedding_coumn = tf.feature_column.embedding_column(
        categorical_column=category_coumn,
        dimension=4
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=[terms_embedding_coumn],
        optimizer=optimizer,
        hidden_units=[10,10],
    )
    classifier.train(input_fn=lambda : _input_fn([prefix+train_file]), steps=5000)
    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+train_file]),
                                             steps=1000)

    print ("------------embedding all terms------")
    print("training set metrics :")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")

    evaluation_metrics = classifier.evaluate(input_fn=lambda : _input_fn([prefix+test_file]),
                                             steps=1000)
    print("test set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
    print ("---")
    return classifier


classifier = dnnEmbeddingWithMoreTerms()
