

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess) 

from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, Input, Concatenate, Add
from keras.callbacks import EarlyStopping, CSVLogger

from loader import load_20news
from custom_layer import MonteCarloLRF, SeparableMonteCarloLRF, SeparableMonteCarloMaxPoolingV2, RandomLRF

import numpy as np
import sklearn as sk
import pickle


#
# Arguments
#

import sys
if len(sys.argv) != 3:
    print('wrong args')
neib = int(sys.argv[1])
drop = float(sys.argv[2])

#
# converted from ipynb
#

top_words=10000
sparse=False
remove_short_documents=True
notebook = 'mcNet_top10k_temptative_42'



(input_shape, nb_classes), (X_train, X_test, Y_train, Y_test), graph_data =     load_20news(data_home='data', top_words=top_words, sparse=sparse,
                remove_short_documents=remove_short_documents, verbose=False)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
num_classes = Y_train.shape[1]



# In[8]:

METRIC = 'cosine'#'euclidean'
distances = sk.metrics.pairwise.pairwise_distances(graph_data, metric=METRIC, n_jobs=-2)

# enforce exact zero
for k in range(distances.shape[0]):
  distances[k,k] = 0.

# max normalize
#distances /= distances.max()
distances /= distances.max(axis=1).reshape((distances.shape[0], 1))

# use tricube kernel (becaause of flatness around 0)
probabilities = (1. - np.abs(distances) ** 3) ** 3

# remove auto connections (which are taken anyway in LRF)
for k in range(probabilities.shape[0]):
  probabilities[k,k] = 0.

# normalize proba
probabilities /= np.sum(probabilities, axis=1).reshape((probabilities.shape[0], 1))

# pickled for later use
#pickle.dump(probabilities, open(path,"wb")) 

# In[9]:


if False:
  probabilities = np.ones(probabilities.shape)

  # remove auto connections (which are taken anyway in LRF)
  for k in range(probabilities.shape[0]):
    probabilities[k,k] = 0.

  # renormalize proba
  probabilities /= np.sum(probabilities, axis=-1).reshape((probabilities.shape[0], 1))
  assert ((np.sum(probabilities, axis=-1) - 1) < 0.000001).all()

# In[10]:


batch_size = 64
X = Input(shape=(X_train.shape[1], X_train.shape[2]))
H1 = Dropout(drop)(X)

H = SeparableMonteCarloLRF(probabilities, LRF_size=neib, activation='relu')(H1)
H = Conv1D(32, kernel_size=1, activation='relu', padding='same') (H)
H = Dropout(drop)(H)
#H = Add()([H,X])

H2 = SeparableMonteCarloLRF(probabilities, LRF_size=neib, activation='relu')(H)
H2 = Conv1D(32, kernel_size=1, activation='relu', padding='same') (H)
H2 = Dropout(drop)(H2)

H2 = Conv1D(1, kernel_size=1, activation='relu', padding='same') (H2)
H2 = Add()([H1,H2])


L = Flatten()(H2)
L = Dense(500, activation='relu')(L)
L = Dropout(drop)(L)
Y = Dense(num_classes, activation='softmax')(L)
model = Model(inputs=X, outputs=Y)
#model.summary()


# In[11]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
print(str(score), flush=True)
