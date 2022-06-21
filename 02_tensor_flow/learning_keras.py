import tensorflow as tf
from tensorflow.keras.layers import Layer

class Linear(Layer):
    """ y = x.w + b """

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Instatiate our Layer
linear_layer = Linear(4)

y = linear_layer(tf.ones((2, 3)))

assert len(linear_layer.weights) == 2