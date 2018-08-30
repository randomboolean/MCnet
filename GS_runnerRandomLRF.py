import argparse

parser = argparse.ArgumentParser(description='Training routine for RandomLRF on 20news')

parser.add_argument('--batch-size', '-b', nargs='?', default=128, type=int)
parser.add_argument('--epochs', '-e', nargs='?', default=50, type=int)
parser.add_argument('--dropout', '-d', nargs='?', default=0.5, type=float)
parser.add_argument('--LRF-size', '-l', nargs='?', default=16, type=int)
parser.add_argument('--filters', '-f', nargs='?', default=25, type=int)

args = parser.parse_args()
print(args)

import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger

from loader import load_20news
from custom_layer import MonteCarloLRF, SeparableMonteCarloLRF, SeparableMonteCarloMaxPoolingV2, RandomLRF

import numpy as np
import sklearn as sk
import pickle

top_words=10000
sparse=False
remove_short_documents=True
#notebook = 'mcNet_top10k_temptative_42'

# preparing data
(input_shape, nb_classes), (X_train, X_test, Y_train, Y_test), graph_data = \
    load_20news(data_home='data', top_words=top_words, sparse=sparse,
                remove_short_documents=remove_short_documents, verbose=False)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
num_classes = Y_train.shape[1]

#%%time
#Process next cell only once
path = os.path.join('probabilities_' + 
                    'top' + str(top_words) +
                    '_sparse' + str(sparse) +
                    '_removeShorts' + str(remove_short_documents) +
                    '.pkl')
if os.path.isfile(path):
  probabilities = pickle.load(open(path, "rb"), encoding='latin1')

if not(os.path.isfile(path)):
  METRIC = 'cosine'#'euclidean'
  distances = sk.metrics.pairwise.pairwise_distances(graph_data, metric=METRIC, n_jobs=-2)

  # enforce exact zero
  for k in xrange(distances.shape[0]):
    distances[k,k] = 0.

  # max normalize
  #distances /= distances.max()
  distances /= distances.max(axis=1).reshape((distances.shape[0], 1))

  # use tricube kernel (becaause of flatness around 0)
  probabilities = (1. - np.abs(distances) ** 3) ** 3

  # remove auto connections (which are taken anyway in LRF)
  for k in xrange(probabilities.shape[0]):
    probabilities[k,k] = 0.

  # normalize proba
  probabilities /= np.sum(probabilities, axis=1).reshape((probabilities.shape[0], 1))
  
  # pickled for later use
  pickle.dump(probabilities, open(path,"wb"))
  
model = Sequential()
model.add(RandomLRF(probabilities, LRF_size=args.LRF_size, filters=args.filters, activation='relu',
                       input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Flatten())
model.add(Dropout(args.dropout))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopper = EarlyStopping(min_delta=0.001, patience=2)
#Namespace(LRF_size=16, batch_size=128, dropout=0.5, epochs=50, filters=25)
csv = CSVLogger(str(args.LRF_size) + '_' + str(args.batch_size) + '_' + str(args.dropout) + '_' + str(args.filters) + '_log.csv')

history = model.fit(X_train, Y_train,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    verbose=1,
                    callbacks=[early_stopper, csv],
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
print(score[1])
sys.stdout.flush()