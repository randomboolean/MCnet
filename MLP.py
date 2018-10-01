

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

#
# converted from ipynb
#

top_words=10000
sparse=False
remove_short_documents=True

(input_shape, nb_classes), (X_train, X_test, Y_train, Y_test), graph_data =     load_20news(data_home='data', top_words=top_words, sparse=sparse,
                remove_short_documents=remove_short_documents, verbose=False)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]* X_test.shape[2])
num_classes = Y_train.shape[1]


model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# In[11]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

#print('Test loss:', score[0])
print('Test accuracy:' + str(score[1]))
