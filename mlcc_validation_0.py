#https://colab.sandbox.google.com/notebooks/mlcc/validation.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation-colab&hl=en#scrollTo=_xSYTarykO8U
import tensorflow as tf
import os
import urllib
import pandas as pd
import numpy as np
import math
from sklearn import metrics
from matplotlib import pyplot as plt


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.max_columns = 20


pd.options.display.float_format = '{:.1f}'.format

DATA_URL = "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
data_file = DATA_URL.split("/")[-1]
TEST_URL = "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv"
test_file = TEST_URL.split("/")[-1]

label_name = "median_house_value"

def downloadFile():
    '''
    Download the file if file is not downloaded
    :return:
    '''
    if not os.path.exists(data_file):
        raw = urllib.urlopen(DATA_URL).read()
        with open(data_file, "w") as df:
            df.write(raw)

    if not os.path.exists(test_file):
        raw = urllib.urlopen(TEST_URL).read()
        with open(test_file, "w") as df:
            df.write(raw)

def loadPdData():
    '''
    load the data as pandas data frame
    :return: data frame
    '''
    data = pd.read_csv(data_file)
    data[label_name] /=1000.0
    data = data.reindex(np.random.permutation(data.index))
    return data

downloadFile()
california_housing_dataframe = loadPdData()


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
    return processed_features

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = pd.DataFrame()
training_targets[label_name]= california_housing_dataframe[label_name].head(12000)

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = pd.DataFrame()
validation_targets[label_name] = california_housing_dataframe[label_name].tail(5000)

'''
plt.figure(figsize=(13,8))
ax=plt.subplot(1, 2,1)
ax.set_autoscalex_on(False)
ax.set_xlim(-126, -112)

ax.set_autoscaley_on(False)
ax.set_ylim(32, 43)
plt.scatter(validation_examples["longitude"], validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"]/validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()
plt.show()
'''
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = tf.data.Dataset.from_tensor_slices((features, targets)).batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)
    return ds.make_one_shot_iterator().get_next()

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(key) for key in input_features])


california_housing_test_data = pd.read_csv(test_file)
california_housing_test_data[label_name] /=1000.0
test_examples = preprocess_features(california_housing_test_data)
test_targets = california_housing_test_data[label_name].values

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    feature_columns = construct_feature_columns(training_examples)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer,  5.0)
    regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                             optimizer = optimizer)
    periods = 10
    steps_per_period = steps/periods
    training_input_fn = lambda : my_input_fn(training_examples, training_targets, batch_size=batch_size, shuffle=True)
    predict_training_input_fn = lambda : my_input_fn(training_examples, training_targets, batch_size=1, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, batch_size=1, shuffle=False, num_epochs=1)

    training_rmse = []
    validation_rmse =[]
    for period in range(0, periods):
        _ = regressor.train(input_fn=training_input_fn, steps=steps_per_period )
        training_predictions = regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = [item["predictions"][0] for item in training_predictions]

        validation_predictions = regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = [item["predictions"][0] for item in validation_predictions]

        training_mse = metrics.mean_squared_error(training_predictions, training_targets[label_name].values)
        training_rmse.append(math.sqrt(training_mse))

        validation_mse = metrics.mean_squared_error(validation_predictions, validation_targets[label_name].values)
        validation_rmse.append(math.sqrt(validation_mse))

        print("iteration: {:d}, Training RMSE: {:.3f}. Validation RMSE: {:.3f}".format(period, training_rmse[-1], validation_rmse[-1]))

    plt.figure(figsize=(13,8))
    plt.xlabel("period")
    plt.ylabel("rmse")
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return regressor


regressor = train_model(
    # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
    learning_rate=0.00003,
    steps=300,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

test_input_fn = lambda : my_input_fn(test_examples, test_targets, shuffle=False, num_epochs=1)
test_predictions = regressor.predict(input_fn=test_input_fn)
test_predictions = [item["predictions"][0] for item in test_predictions]
mse = metrics.mean_squared_error(test_predictions, test_targets)
rmse = math.sqrt(mse)
print "loss for test data: {:.3f}".format(rmse)

