import argparse
import json
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2 
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

TRAINING_PATH = 'data/numerai_training_data.csv'
TESTING_PATH = 'data/numerai_tournament_data.csv'

CONFIG_PATH = 'data/model.json'
OUTPUT_PATH = 'data/submission.csv'
    
def load_data(input_file = TRAINING_PATH):
    df = pd.read_csv(input_file)
    x_train = df.iloc[:,0:-1]
    y_train = df.iloc[:,-1]    
    return x_train, y_train

def save_predictions(model, output_file=OUTPUT_PATH, input_file = TESTING_PATH):
    df = pd.read_csv(input_file)
    predictions = model.predict(np.array(df.iloc[:,1:]))
    output = pd.DataFrame(predictions.flatten())
    output.index = df.iloc[:,0]
    output.columns = ['probability']
    output.to_csv(output_file)
    return output

def perform_cross_validation(model_config, x, y, k_folds, loss='binary_crossentropy'):
    skf = StratifiedKFold(y, k_folds)
    error = 0.0
    for train_idx, test_idx in skf:
        x_train = x.ix[train_idx]
        y_train = y.ix[train_idx]
        x_test = x.ix[test_idx]
        y_test = y.ix[test_idx]
        model = model_from_json(model_config)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(np.array(x_train), np.array(y_train), nb_epoch=50, batch_size=1000, verbose=0)
        predictions = model.predict(np.array(x_test))
        error += log_loss(np.array(y_test), predictions.flatten())
    cv_error = error/k_folds
    return cv_error
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='+', type=int, required=True)
    parser.add_argument('-r', nargs='+', type=float, required=True)
    parser.add_argument('-k', type=int, required=False, default=5)
    results = parser.parse_args()

    n_hidden = results.n
    regularization_param = results.r
    k_folds = results.k

    x_train, y_train = load_data()
    n_features = x_train.shape[1]

    best_cv_error = 10000
    
    models_to_try = len(n_hidden) * len(regularization_param)
    ctr = 0
    
    for n in n_hidden:
        for r in regularization_param:
            ctr +=1 
            print "Evaluating model %d of %d" % (ctr, models_to_try)
            model = Sequential()
            model.add(Dense(n, input_dim=n_features, activation='sigmoid', W_regularizer=l2(r)))
            model.add(Dense(1, activation='sigmoid', W_regularizer=l2(r)))
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            cv_error = perform_cross_validation(model.to_json(), x_train, y_train, k_folds)
            if cv_error < best_cv_error:
                best_cv_error = cv_error
                best_model_config = model.to_json()
            
    print "===================="
    print "Training final model"

    final_model = model_from_json(best_model_config)
    final_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    final_model.fit(np.array(x_train), np.array(y_train), nb_epoch=50, batch_size=1000, verbose=0)
    
    predictions = save_predictions(final_model)
    
    with open(CONFIG_PATH, "w") as f:
        f.write(json.dumps(final_model.to_json(), ensure_ascii=False))
    f.close()
    
        