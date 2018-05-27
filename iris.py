from __future__ import print_function
import tensorflow as tf


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

FEATURES = ['SepalLength', 'SepalWidth',
            'PetalLength', 'PetalWidth']
LABEL= 'Species'

tf.logging.set_verbosity(tf.logging.INFO)
def download_data():
  train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split("/")[-1],
                                       origin=TRAIN_URL)

  test_path = tf.keras.utils.get_file(fname=TEST_URL.split("/")[-1],
                                      origin=TEST_URL)
  return (train_path, test_path)


def create_feature_column():
  feature_columns = []
  for key in FEATURES:
    feature_columns.append(tf.feature_column.numeric_column(key, shape=1))

  return feature_columns

def input_fn(filename, batch_size, is_training):
  def _parse_csv(one_line):
    decoded = tf.decode_csv(one_line, record_defaults=[[]]*5)
    features= dict(zip(FEATURES, decoded[:-1]))
    label = tf.cast(decoded[-1], tf.int32)
    return features, label

  def _input_fn():
    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.skip(1).map(_parse_csv)
    if (is_training) :
      dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

  return _input_fn




######################run train here
train_file, test_file = download_data()

classifier = tf.estimator.DNNClassifier(feature_columns=create_feature_column(),
                                        hidden_units=[10, 3], n_classes=3)

train_input_fn = input_fn(train_file, 32, True)
classifier.train( input_fn=train_input_fn, steps=400)

test_input_fn = input_fn(test_file, 32, False)
evaluation_result = classifier.evaluate(test_input_fn)
print (evaluation_result)
