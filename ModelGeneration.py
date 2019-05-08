from FeaGeneration import GetTrainData, GetTestData

from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from keras.models import model_from_json
import numpy as np


def main():
    a, b = GetTrainData()
    c,d = GetTestData()
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    le = preprocessing.LabelEncoder()
    le.fit(b)
    np.save('classes.npy', le.classes_)
    train_label = le.transform(b)
    test_label = le.transform(d)
    x_train = a
    x_test = c
    y_train = train_label
    y_test = test_label
    #测试全连接网络
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3, 512)),
    keras.layers.Dense(256),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)
    fully_connected_pred = model.predict_classes(x_test)
    report_fc = classification_report(y_test, fully_connected_pred)
    print(report_fc)# serialize model to JSON

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    return()

main()
# later...
