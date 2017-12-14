from keras.layers import Layer
import numpy as np

class MonteCarloLRF(Layer):

  def __init__(self,
              probabilities,
              LRF_size,

              use_bias = True,
              activation = None,
              kernel_initializer = 'he_uniform', kernel_regularizer = None, kernel_constraint = None,
              bias_initializer = 'zeros', bias_regularizer = None, bias_constraint = None,
              activity_regularizer = None,
              **kwargs):

    super(MonteCarloLRF, self).__init__(**kwargs)
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

    # fill up LRF_getter
    self.LRF_getter = np.zeros(shape=(self.n, self.p, self.LRF_size))
    for channel in xrange(self.p):
      for i in xrange(self.n):
        self.LRF_getter[i, channel, 0] = i
        self.LRF_getter[i, channel, 1:] = np.random.choice(np.arange(self.n), 
                                                  size=(self.LRF_size - 1,),
                                                  replace=False,
                                                  p=self.probabilities[i,:])
  def build(self, input_shape):
    self.b, self.n, self.p = input_shape
    self.build_LRF()
    self.kernel = self.add_weight(name='{}_W'.format(self.name),
                                    shape=(self.k, self.p),
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
    super(MonteCarloLRF, self).build(input_shape)
    self.input_spec = InputSpec(ndim=3)
    self.built = True

  def call(self, x):
    # gather LRF, x(b,n,p), LRF_getter(n,p,k)
    col = tf.gather(x, self.LRF_getter, axis=[1,2]) #not sure about this part
    print col.shape #to remove

    # col(b,n,p,k) (dot) kernel(k,p) -> y(b,n,p)
    col = tf.transpose(col, [0,1,3,2])
    y = col * tf.reshape(self.kernel, [1,1,self.LRF_size,self.p])
    y = tf.reduce_sum(y, axis=2)

    if self.use_bias:
      y += tf.reshape(self.bias, (1, 1, self.q))
    return self.activation(y)