# Source code of FrangiLayer
# Authored by X. X.

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Sequential, Model
import tensorflow_addons as tfa



class Scaling01(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Scaling01, self).__init__(**kwargs)
    def call(self, inputs):
        min_value = tf.math.reduce_min(inputs)
        max_value = tf.math.reduce_max(inputs)
        return_tensor = tf.math.divide(tf.math.subtract(inputs, min_value),
                                       max_value - min_value)
        #print(max_value)
        #print(min_value)
        return return_tensor

    def get_config(self):
        config = super(Scaling01, self).get_config()
        return config

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
      super(ScaleLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        kernel_constraint_scale = tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1)
        initializer = tf.keras.initializers.Ones()

        self.scale = self.add_weight(name="scale",
                                     shape=(1, 1), initializer=initializer, trainable=True,
                                     constraint=kernel_constraint_scale
                                     )

    def call(self, inputs):
      return inputs * self.scale[0,0]

    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        return config


class FrangiLayer(keras.layers.Layer):
    def __init__(self, sigma=1, **kwargs):
        self.sigma = sigma
        super(FrangiLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        kernel_constraint_beta = tf.keras.constraints.MinMaxNorm(
            min_value=0.05, max_value=10.0, rate=1.0, axis=0
        )
        kernel_constraint_gamma = tf.keras.constraints.MinMaxNorm(
            min_value=0.05, max_value=50, rate=1.0, axis=0
        )

        my_initializer_beta = tf.keras.initializers.RandomUniform(minval=0.05, maxval=10, seed=None)
        my_initializer_gamma = tf.keras.initializers.RandomUniform(minval=0.05, maxval=50, seed=None)

        self.beta = self.add_weight(name="beta",
            shape=(1,1), initializer=my_initializer_beta, trainable=True, constraint=kernel_constraint_beta)
        self.gamma = self.add_weight(name="gamma",
                                         shape=(1, 1), initializer=my_initializer_gamma, trainable=True,
                                         constraint=kernel_constraint_gamma)

    def get_config(self):
        config = super(FrangiLayer, self).get_config()
        config.update({
            'sigma': self.sigma
        })
        return config

    #Hessian calculation
    def hessian_matrix(self, inputs, sigma):
        #size of the gaussian kernel (see scipy implementation)
        truncate = 4
        filter_shape = 2 * int(truncate * sigma + 0.5) + 1

        ggg = tfa.image.gaussian_filter2d(image=inputs,
                                          filter_shape=filter_shape,
                                          padding= "REFLECT",#"SYMMETRIC",
                                          sigma=sigma)
        #first partial derivative
        ggg = tf.image.image_gradients(ggg)
        #second partial derivative
        gggx = tf.image.image_gradients(ggg[0])
        gggy = tf.image.image_gradients(ggg[1])
        return [gggx[0], gggx[1], gggy[0], gggy[1]]

    def calculate_frangi_sigma(self, inputs, sigma, beta, gamma):
        gamma_2 = gamma * gamma
        beta_2 = beta * beta
        hm = self.hessian_matrix(inputs, sigma)
        # https://sci-hub.se/10.1007/bfb0056195
        #rescalling by sigma^2
        for a in range(len(hm)):
            hm[a] = tf.math.scalar_mul(sigma * sigma, hm[a])

        #second-order partial derivatives
        M00 = hm[0]
        M01 = hm[1]  # hm[1] == hm[2]; M01 == M10
        M11 = hm[3]
        # https://github.com/solivr/frangi_filter/blob/master/frangiFilter2D.py

        #calculate eigenvalues of Hessian matrix in every point
        l_left = (M00 + M11) / 2.0001
        l_right = 4.0 * tf.math.multiply(M01, M01)
        Mdiff = M00 - M11
        l_right = l_right + tf.math.multiply(Mdiff, Mdiff)
        l_right = tf.math.sqrt(l_right) / 2.0001
        l1 = l_left + l_right
        l2 = l_left - l_right

        #oder eigenvalues by ascending module
        l1g = tf.greater(tf.math.abs(l1), tf.math.abs(l2))
        l2_sorted = tf.where(l1g, l1, l2)
        l1_sorted = tf.where(l1g, l2, l1)

        l1 = l1_sorted
        l2 = l2_sorted

        #replace zeros by small values
        l2_divider = tf.where(tf.equal(l2, 0), 0.00000001, l2)

        #calculate RB
        RB = tf.math.divide(tf.math.abs(l1), tf.math.abs(l2_divider))
        # https://www.sciencedirect.com/science/article/pii/S2213597920300409
        RB_2 = tf.math.multiply(RB, RB)
        #calculate S
        S_2 = tf.math.multiply(l1, l1) + tf.math.multiply(l2, l2)  # without sqrt we get power 2 of S

        #calcylate Frangi filter
        exp_line = tf.math.exp(-tf.math.divide(RB_2, 2.00001 * beta_2))
        exp_back = 1.0 - (tf.math.exp(-tf.math.divide(S_2, 2.0001 * gamma_2)))
        VS = tf.math.multiply(exp_line, exp_back)
        l2g0 = tf.greater(l2, 0)
        l2g0 = tf.cast(l2g0, tf.bool)
        VS = tf.where(l2g0, 0.0, VS)
        return VS

    def call(self, inputs):
        sigmas = [self.sigma]
        beta = float(self.beta[0,0])
        gamma = float(self.gamma[0,0])
        VS = None
        #iterate by all sigmas
        for sigma in sigmas:
            #sigma = 4
            if VS is None:
                VS = self.calculate_frangi_sigma(inputs, sigma, beta, gamma)
            else:
                VS1 = self.calculate_frangi_sigma(inputs, sigma, beta, gamma)
                #get maximum of filters
                VS = tf.math.maximum(VS, VS1)
        return VS
