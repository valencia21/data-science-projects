from __future__ import print_function

import numpy as np
import pandas as pd

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    training = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/training_preprocessed")
    validation = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/validation_preprocessed")
    test = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/test_preprocessed")
    x_train = training.drop(columns="y")
    y_train = training["y"]
    x_test = validation.drop(columns="y")
    y_test = validation["y"]
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss="mse", metrics=["mae"])
    
    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128, 256])}},
              epochs={{choice([30, 50, 100])}},
              verbose=2,
              validation_split=0.1)
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                              data=data(),
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
    x_train, y_train, x_test, y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)