from __future__ import print_function
import tensorflow as tf
import os
import urllib


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

FEATURES = ['SepalLength', 'SepalWidth',
            'PetalLength', 'PetalWidth']
LABEL= 'Species'

tf.logging.set_verbosity(tf.logging.INFO)


def maydownload():
  trainfile = TRAIN_URL.split("/")[-1]
  testfile = TEST_URL.split("/")[-1]

  if not os.path.exists(trainfile):
   raw = urllib.urlopen(TRAIN_URL).read()
   with open(trainfile, 'w') as f:
     f.write(raw)

  if not os.path.exists(testfile):
    raw = urllib.urlopen(TEST_URL).read()
    with open(testfile, 'w') as f:
      f.write(raw)

  return trainfile, testfile

def input_fn(filename, batch_size, is_training):

  def _parse_csv(line):
    parsed = tf.decode_csv(line, record_defaults=[[]]*(len(FEATURES)+1))
    features = dict(zip(FEATURES, parsed[:-1]))
    label = tf.cast(parsed[-1], tf.int32)
    return features, label

  def _input_fn():
    dataset = tf.data.TextLineDataset([filename]).skip(1).map(_parse_csv)
    if is_training:
      dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
  return _input_fn

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainfile, testfile = maydownload()
  feature_columns = [ tf.feature_column.numeric_column(key, shape=1) for key in FEATURES]
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10,20,10],
                                          n_classes=3
                                          )
  classifier.train(input_fn=input_fn(trainfile, 32, True), steps=400)

  evaluation = classifier.evaluate(input_fn=input_fn(testfile, 32, False))
  print(evaluation)
  print ('Accuracy (tensorflow): {0:f}'.format(evaluation['accuracy']))

  predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
  }

  def predict_input_fn():
    return tf.data.Dataset.from_tensor_slices(predict_x).batch(32).make_one_shot_iterator().get_next()

  predictions = classifier.predict(input_fn=predict_input_fn)
  for p in predictions:
    class_id = p['class_ids'][0]
    probability = p['probabilities'][class_id]
    print (class_id)


if __name__ == '__main__':
  tf.app.run()



