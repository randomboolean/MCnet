import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints

class SeparableMonteCarloLRF(Layer):

  def __init__(self,
              probabilities,
              LRF_size,

              use_bias = True,
              activation = None,
              kernel_initializer = 'he_uniform', kernel_regularizer = None, kernel_constraint = None,
              bias_initializer = 'zeros', bias_regularizer = None, bias_constraint = None,
              activity_regularizer = None,
              **kwargs):

    super(SeparableMonteCarloLRF, self).__init__(**kwargs)
    self.probabilities = probabilities
    self.LRF_size = LRF_size

    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.use_bias = use_bias
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.activity_regularizer = regularizers.get(activity_regularizer)

  def compute_output_shape(self, input_shape):
    return input_shape

  def build_LRF(self):
    # check they are probabilites and diag is 0
    assert not((np.sum(self.probabilities, axis=-1) - 1).all())
    assert np.diag(self.probabilities).sum() == 0.

    # fill up LRF_getter to be used as indices by tf.gather_nd
    self.LRF_getter = np.zeros(shape=(self.n, self.p, self.LRF_size, 2), dtype='int32')
    for channel in xrange(self.p):
      for i in xrange(self.n):
        self.LRF_getter[i, channel, 0, 0] = i
        self.LRF_getter[i, channel, 1:, 0] = np.random.choice(np.arange(self.n), 
                                                  size=(self.LRF_size - 1,),
                                                  replace=False,
                                                  p=self.probabilities[i,:])
        for j in xrange(self.LRF_size):
          self.LRF_getter[i, channel, j, 1] = channel
  def build(self, input_shape):
    self.b, self.n, self.p = input_shape
    self.build_LRF()
    self.kernel = self.add_weight(name='{}_W'.format(self.name),
                                    shape=(self.LRF_size, self.p),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True)
    self.bias = self.add_weight(name='{}_b'.format(self.name),
                                    shape=(self.p,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True)
    super(SeparableMonteCarloLRF, self).build(input_shape)
    self.input_spec = InputSpec(ndim=3)
    self.built = True

  def call(self, x):
    # gather LRF, x(b,n,p), LRF_getter(n,p,LRF_size,2)
    col = tf.gather_nd(tf.transpose(x, [1,2,0]), self.LRF_getter)
    col = tf.transpose(col, [3,0,2,1])

    # col(b,n,LRF_size,p) (dot) kernel(LRF_size,p) -> y(b,n,p)
    y = col * tf.reshape(self.kernel, [1,1,self.LRF_size,self.p])
    y = tf.reduce_sum(y, axis=2)

    if self.use_bias:
      y += tf.reshape(self.bias, (1, 1, self.p))
    return self.activation(y)

class MonteCarloLRF(Layer):

  def __init__(self,
              probabilities,
              LRF_size,
              filters,

              use_bias = True,
              activation = None,
              kernel_initializer = 'he_uniform', kernel_regularizer = None, kernel_constraint = None,
              bias_initializer = 'zeros', bias_regularizer = None, bias_constraint = None,
              activity_regularizer = None,
              **kwargs):

    super(MonteCarloLRF, self).__init__(**kwargs)
    self.probabilities = probabilities
    self.LRF_size = LRF_size
    self.q = filters
    
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.use_bias = use_bias
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.activity_regularizer = regularizers.get(activity_regularizer)
  
  def compute_output_shape(self, input_shape):
    self.b, self.n, self.p = input_shape
    return self.b, self.n, self.q

  def build_LRF(self):
    # check they are probabilites and diag is 0
    assert not((np.sum(self.probabilities, axis=-1) - 1).all())
    assert np.diag(self.probabilities).sum() == 0.

    # fill up LRF_getter to be used as indices by tf.gather_nd
    self.LRF_getter = np.zeros(shape=(self.n, self.p, self.q, self.LRF_size, 2), dtype='int32')
    for feature_map in xrange(self.q):
      for channel in xrange(self.p):
        for i in xrange(self.n):
          self.LRF_getter[i, channel, feature_map, 0, 0] = i
          self.LRF_getter[i, channel, feature_map, 1:, 0] = np.random.choice(np.arange(self.n), 
                                                                  size=(self.LRF_size - 1,),
                                                                  replace=False,
                                                                  p=self.probabilities[i,:])
          for j in xrange(self.LRF_size):
            self.LRF_getter[i, channel, feature_map, j, 1] = channel
  
  def build(self, input_shape):
    self.b, self.n, self.p = input_shape
    self.build_LRF()
    self.kernel = self.add_weight(name='{}_W'.format(self.name),
                                    shape=(self.LRF_size, self.p, self.q),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True)
    self.bias = self.add_weight(name='{}_b'.format(self.name),
                                    shape=(self.q,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True)
    super(MonteCarloLRF, self).build(input_shape)
    self.input_spec = InputSpec(ndim=3)
    self.built = True

  def call(self, x):
    # gather LRF, x(b,n,p), LRF_getter(n,p,q,LRF_size,2) -> col(b,n,LRF_size,p,q)
    col = tf.gather_nd(tf.transpose(x, [1,2,0]), self.LRF_getter)
    col = tf.transpose(col, [4,0,3,1,2])

    # col(b,n,LRF_size,p,q) (dot) kernel(LRF_size,p,q) -> y(b,n,q)
    y = col * tf.reshape(self.kernel, [1,1,self.LRF_size,self.p,self.q])
    y = tf.reduce_sum(y, axis=(2,3))

    if self.use_bias:
      y += tf.reshape(self.bias, (1, 1, self.q))
    return self.activation(y)