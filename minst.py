import tensorflow as tf
import os
import glob

from matplotlib import pyplot  as plt
from matplotlib import gridspec
from matplotlib import cm
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#wget https://storage.googleapis.com/mledu-datasets/mnist_train_small.csv -O /tmp/mnist_train_small.csv
data = pd.read_csv("mnist_train_small.csv", header=None)
data = data.head(10000)
data = data.reindex(np.random.permutation(data.index))

test = pd.read_csv("mnist_test.csv", header=None)

#print data.head()

def extractLabelAndFeatures(data):
    label = data[0]
    features = data.iloc[:, 1:]
    return features, label

train_features, train_label = extractLabelAndFeatures(data[:7500])
validation_fearues, validation_label = extractLabelAndFeatures(data[7500:-1])
test_features, test_label =extractLabelAndFeatures(test)
#print validation_fearues.head()

'''
_, ax = plt.subplots()
ax.matshow( validation_fearues.iloc[0, :].values.reshape(28, 28))
ax.set_title("letter")
plt.show()
'''

#fearture scaling
train_features /=255
validation_fearues /=255


def getFeatureColumns():
    return set([tf.feature_column.numeric_column("pixels", shape=784)])


def create_training_input_fn(features, label, batch_size=1, shuffle=True, num_epoch=None):

    def _input_fn(num_epoch=None, shuffle=True):
        idx = np.random.permutation(features.index)
        features1 = {"pixels": features.reindex(idx)}
        label1 = np.array(label[idx])

        ds = tf.data.Dataset.from_tensor_slices((features1, label1))
        ds = ds.batch(batch_size).repeat(num_epoch)

        if (shuffle):
            ds = ds.shuffle(10000)
        return ds.make_one_shot_iterator().get_next();

    return _input_fn

def create_predict_input_fn(features, labels, batch_size):

    def _input_fn():
        features1 = {"pixels": features}
        labels1 = np.array(labels)

        ds = tf.data.Dataset.from_tensor_slices( (features1, labels1)).batch(batch_size)
        return ds.make_one_shot_iterator().get_next()
    return _input_fn

def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    feature_columns = getFeatureColumns()

    periods = 10
    period_step = steps/periods

    train_loss = []
    validation_loss = []

    train_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
    train_predict_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    validation_predict_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    estimator = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        optimizer=optimizer
    )

    for period in range(periods):
        estimator.train(
            input_fn=train_input_fn,
            steps = period_step
        )

        #predict training sample
        traning_predicitons= list(estimator.predict(input_fn=train_predict_fn))
        training_probabilities = np.array([item["probabilities"] for item in traning_predicitons])
        training_class_ids =     np.array([item['class_ids'][0] for item in traning_predicitons])
        training_one_hot = tf.keras.utils.to_categorical(training_class_ids, 10)

        validation_predictions= list(estimator.predict(input_fn=validation_predict_fn))
        validation_probabilities = np.array( [item["probabilities"] for item in validation_predictions])
        validation_class_ids = np.array( [item["class_ids"][0] for item in validation_predictions])
        validation_one_hot = tf.keras.utils.to_categorical(validation_class_ids, 10)

        train_log_loss = metrics.log_loss(training_targets, training_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_one_hot)
        print "peroid %d, %0.2f, %0.2f"%(period, train_log_loss, validation_log_loss)
        train_loss.append(train_log_loss)
        validation_loss.append(validation_log_loss)
    print "traning finished"

    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(estimator.model_dir, 'events.out.tfevents*')))

    final_predctions = estimator.predict(input_fn=validation_predict_fn)
    final_predctions = np.array([ item["class_ids"][0] for item in final_predctions])

    accuracy = metrics.accuracy_score(validation_targets, final_predctions)
    print (accuracy)

    plt.xlabel("period")
    plt.ylabel("log loss")
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.legend()
    plt.show()
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predctions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    return estimator;
'''
classifier = train_linear_classification_model(
    learning_rate=0.01,
    steps=200,
    batch_size=100,
    training_examples=train_features,
    training_targets=train_label,
    validation_examples=validation_fearues,
    validation_targets=validation_label)
'''

def train_dnn_classification_model(
        learning_rate,
        steps,
        hidden_units,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    feature_columns = getFeatureColumns()

    periods = 10
    period_step = steps/periods

    train_loss = []
    validation_loss = []

    train_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
    train_predict_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    validation_predict_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=10,
        optimizer=optimizer
    )

    for period in range(periods):
        estimator.train(
            input_fn=train_input_fn,
            steps = period_step
        )

        #predict training sample
        traning_predicitons= list(estimator.predict(input_fn=train_predict_fn))
        training_probabilities = np.array([item["probabilities"] for item in traning_predicitons])
        training_class_ids =     np.array([item['class_ids'][0] for item in traning_predicitons])
        training_one_hot = tf.keras.utils.to_categorical(training_class_ids, 10)

        validation_predictions= list(estimator.predict(input_fn=validation_predict_fn))
        validation_probabilities = np.array( [item["probabilities"] for item in validation_predictions])
        validation_class_ids = np.array( [item["class_ids"][0] for item in validation_predictions])
        validation_one_hot = tf.keras.utils.to_categorical(validation_class_ids, 10)

        train_log_loss = metrics.log_loss(training_targets, training_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_one_hot)
        print "peroid %d, %0.2f, %0.2f"%(period, train_log_loss, validation_log_loss)
        train_loss.append(train_log_loss)
        validation_loss.append(validation_log_loss)
    print "traning finished"

    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(estimator.model_dir, 'events.out.tfevents*')))

    final_predctions = estimator.predict(input_fn=validation_predict_fn)
    final_predctions = np.array([ item["class_ids"][0] for item in final_predctions])

    accuracy = metrics.accuracy_score(validation_targets, final_predctions)
    print (accuracy)

    plt.xlabel("period")
    plt.ylabel("log loss")
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.legend()
    plt.show()
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predctions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return estimator

classifier = train_dnn_classification_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=100,
    hidden_units=[100, 100],
    training_examples=train_features,
    training_targets=train_label,
    validation_examples=validation_fearues,
    validation_targets=validation_label)

test_prediction = classifier.predict(input_fn=create_predict_input_fn(test_features, test_label, batch_size=100))
test_classes = np.array([item["class_ids"][0]] for item in test_prediction)
accurary = metrics.accuracy_score(test_label, test_classes)

print "test accuracy %0.5f"%accurary

