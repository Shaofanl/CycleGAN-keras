from keras.engine.topology import Layer
import keras.backend as K

class InstanceNormalization2D(Layer):
    ''' Thanks for github.com/jayanthkoushik/neural-style '''
    def __init__(self, **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[1],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[1],), initializer="zero", trainable=True)
        super(InstanceNormalization2D, self).build(input_shape)

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            T = K.theano.tensor
            hw = T.cast(x.shape[2] * x.shape[3], K.theano.config.floatX)
            mu = x.sum(axis=-1).sum(axis=-1) / hw
            mu_vec = mu.dimshuffle(0, 1, "x", "x")
            sig2 = T.square(x - mu_vec).sum(axis=-1).sum(axis=-1) / hw
            y = (x - mu_vec) / T.sqrt(sig2.dimshuffle(0, 1, "x", "x") + K.epsilon())
            return self.scale.dimshuffle("x", 0, "x", "x") * y + self.shift.dimshuffle("x", 0, "x", "x")
        else:
            raise NotImplemented("Please complete `CycGAN/layers/padding.py` to run on backend {}.".format(K.backend()))

    def compute_output_shape(self, input_shape):
        return input_shape 
