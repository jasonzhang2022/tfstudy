#ref: https://colab.sandbox.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=firststeps-colab&hl=en#scrollTo=7G12E76-339G
from __future__ import print_function
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
features_names = ["total_rooms"]
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

def loadPdData():
    '''
    load the data as pandas data frame
    :return: data frame
    '''
    data = pd.read_csv(data_file)
    data[label_name] /=1000.0
    return data

def my_input(pd_data, epoch=None, shuffle=True, batch_size=1):

    #features needs to be a dict:
    # the key is feature name. The value is an array
    features = {key: value.values for key, value in pd_data.items() if key in features_names}
    label = pd_data[label_name].values

    dataset = tf.data.Dataset.from_tensor_slices((features, label)).batch(batch_size).repeat(epoch)

    if shuffle:
        dataset = dataset.shuffle(10000)

    return dataset.make_one_shot_iterator().get_next()

def main(unusedargs):

    feature_columns = [tf.feature_column.numeric_column(key=key) for key in features_names]
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                             optimizer=optimizer)
    data = loadPdData()
    _ = regressor.train(input_fn=lambda : my_input(data), steps=100)

    predictions = regressor.predict(input_fn=lambda : my_input(data, epoch=1, shuffle=False))
    predictions = [item["predictions"][0] for item in predictions]
    predictions = np.asanyarray(predictions, dtype=np.float64)
    targets = data[label_name].values

    #dump basic information
    calibrations = pd.DataFrame()
    calibrations["predictions"] = pd.Series(predictions)
    calibrations["targets"] = data[label_name]
    print(calibrations.describe())

    min_value = data[label_name].min()
    max_value = data[label_name].max()
    diff_value = max_value - min_value
    mse=metrics.mean_squared_error(predictions, targets)
    rmse = math.sqrt(mse)
    print("Median House Value: (min: {:.3f}), (max: {:.3f}), (diff:{:.3f}) (mse: {:.3f}), (rmse: {:.3f})".format(min_value,
          max_value, diff_value, mse, rmse))


    sample = data.sample(n=300)
    x0 = sample["total_rooms"].min()
    x1 = sample["total_rooms"].max()


    weight = regressor.get_variable_value("linear/linear_model/total_rooms/weights")[0]
    bias = regressor.get_variable_value("linear/linear_model/bias_weights")
    y0= x0*weight + bias
    y1 =x1*weight + bias

    plt.plot([x0, x1], [y0, y1], c="r")
    plt.xlabel("total_rooms")
    plt.ylabel("median House value")

    plt.scatter(sample["total_rooms"], sample[label_name])
    plt.show()




if __name__ == '__main__':
    tf.app.run()





