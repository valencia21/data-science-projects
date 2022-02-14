from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperas import optim
from hyperas.distributions import choice, uniform
import pandas as pd

def data():
    training = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/training_preprocessed")
    validation = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/validation_preprocessed")
    test = pd.read_csv("Documents/GitHub/data-science-projects/house-price-prediction/Models/tensorflow/test_preprocessed")
    X_train = training.drop(columns="y")
    y_train = training["y"]
    X_valid = validation.drop(columns="y")
    y_valid = validation["y"]
    return X_train, y_train, X_valid, y_valid

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