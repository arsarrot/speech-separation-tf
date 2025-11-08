# Libraries for across all scripts
import tensorflow as tf
import itertools
import numpy as np
import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Layer
from keras_hub.layers import RotaryEmbedding
from einops import rearrange
import time
import random
from itertools import permutations


@tf.keras.utils.register_keras_serializable()
class DownSample(layers.Layer):
    def __init__(
        self, filters, kernel_size=15, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.convA = layers.Conv1D(filters, kernel_size, dilation_rate=1, strides = 1, padding = 'same')
        self.reluA = tf.keras.layers.LeakyReLU(negative_slope = 0.1)
        self.pool = layers.MaxPool1D(2, 2)
        self.bn2a = layers.BatchNormalization()


    def call(self, x):
        if len(x.shape) == 2:
          x = tf.expand_dims(x, axis=-1)
        x = self.convA(x)
        x = self.bn2a(x)
        x = self.reluA(x)
        p = self.pool(x)
        return x, p
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
        })
        return config



@tf.keras.utils.register_keras_serializable()
class BottleNeck(layers.Layer):
    def __init__(
        self, filters, kernel_size=15, padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.convA = layers.Conv1D(filters, kernel_size, dilation_rate=1, strides = 1, padding = 'same')
        self.bnA = layers.BatchNormalization()
        self.reluA = tf.keras.layers.LeakyReLU(negative_slope = 0.1)

    def call(self, x):
        x = self.convA(x)
        x = self.bnA(x)
        x = self.reluA(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class UpSample(layers.Layer):
    def __init__(
        self, filters, kernel_size=5, padding="same", strides=2, **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.us = layers.UpSampling1D(2)
        self.convA = layers.Conv1DTranspose(filters, kernel_size, strides = 1, padding = 'same', dilation_rate = 1, use_bias = False)
        self.reluA = tf.keras.layers.LeakyReLU(negative_slope = 0.1)
        self.bn2a =  layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
        })
        return config



@tf.keras.utils.register_keras_serializable()
class TransposeLayer(layers.Layer):
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.perm = perm
    def call(self, x):
        return tf.transpose(x, perm=self.perm)
    def get_config(self):
        config = super(TransposeLayer, self).get_config()
        config.update({'perm': self.perm})
        return config


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_hidden, num_heads, seq_len, d_k, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_k = float(d_k)
        self.W_q = layers.Dense(3 * num_heads * num_hidden, use_bias=True)
        self.W_k = layers.Dense(2 * num_heads * num_hidden, use_bias=True)
        self.W_v = layers.Dense(num_heads * num_hidden, use_bias=True)
        self.dropout = layers.Dropout(0.1)
        self.mask = self.get_mask(self.seq_len)
        self.group_norm = layers.GroupNormalization(groups=num_heads, axis=-1)
        self._lambda_init = tf.random.uniform(shape=[1])
        self._lambda = self.add_weight(
            name='lambda',
            shape=[1],
            initializer=tf.keras.initializers.Constant(value=self._lambda_init.numpy()),
            trainable=True
        )

    def get_mask(self, size):
        ones = tf.ones((size, size), dtype=tf.float32)
        mask = tf.linalg.band_part(ones, 0, -1) - tf.linalg.band_part(ones, 0, 0)
        return tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)


    def call(self, query, key, values, dropout=0.1, mask=None, training=None):
        batch_size = tf.shape(query)[0]
        query = self.W_q(query)
        query = tf.reshape(query, (batch_size, self.seq_len, self.num_heads, 3 * self.num_hidden))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = self.W_k(key)
        key = tf.reshape(key, (batch_size, self.seq_len, self.num_heads, 2 * self.num_hidden))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        values = self.W_v(values)
        values = tf.reshape(values, (batch_size, self.seq_len, self.num_heads, self.num_hidden))
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        query_1, query_2, gating_score = tf.split(query, num_or_size_splits=3, axis=-1)
        key_1 = key[:, :, :, :self.num_hidden]
        key_2 = key[:, :, :, self.num_hidden:]
        QK_T_1 = tf.matmul(query_1, tf.transpose(key_1, perm=[0, 1, 3, 2])) / tf.math.sqrt(self.d_k)
        QK_T_2 = tf.matmul(query_2, tf.transpose(key_2, perm=[0, 1, 3, 2])) / tf.math.sqrt(self.d_k)
        QK_T_1_norm = tf.nn.softmax(QK_T_1, axis=-1)
        QK_T_2_norm = tf.nn.softmax(QK_T_2, axis=-1)
        if mask:
            mask_expanded = self.mask
            attention_scores = tf.where(mask_expanded == 1, float('-inf'), attention_scores)

        attention_scores = self.dropout(attention_scores, training=training)
        output = tf.matmul(attention_scores, values)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = self.group_norm(output)
        output = output * (1 - self._lambda_init)
        gating_score = tf.transpose(gating_score, perm=[0, 2, 1, 3])
        output = output * tf.sigmoid(gating_score)
        output = tf.reshape(output, (batch_size, self.seq_len, self.num_heads * self.num_hidden))
        return output
    def get_config(self):
            config = super().get_config()
            config.update({
                "num_hidden": self.num_hidden,
                "num_heads": self.num_heads,
                "seq_len": self.seq_len,
                "d_k": self.d_k
            })
            return config


@tf.keras.utils.register_keras_serializable()
class ConvAttn(keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(ConvAttn, self).__init__(**kwargs)
        self.batch_norm = layers.BatchNormalization()
        self.dense = layers.Dense(out_channels)
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=15,
            padding='same',
            depth_multiplier=1
        )
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs):
        x = self.batch_norm(inputs)
        x = self.dense(x)
        x = tf.nn.silu(x)
        residual = self.depthwise_conv(x)
        x = x + residual
        x = self.dropout(x)
        return x


@tf.keras.utils.register_keras_serializable()
class MultiHeadBlock(layers.Layer):
    def __init__(self, num_hidden, num_heads, seq_len, d_k, **kwargs):
        super().__init__(**kwargs)
        self.pos_enc = RotaryEmbedding(max_wavelength=10000, scaling_factor=1.0, sequence_axis=1, feature_axis=-1)
        self.attn = MultiHeadAttention(num_hidden, num_heads, seq_len, d_k)
        self.act = layers.Activation('sigmoid')
        self.convin = ConvAttn(num_hidden * num_heads)
        self.convin2 = ConvAttn(num_hidden * num_heads)
        self.batchnrm = layers.BatchNormalization()

    def call(self, x):
        u = x
        x_shift, x_pass = tf.split(x, num_or_size_splits=2, axis=-1)
        x_shift_padded = tf.pad(x_shift, [[0, 0], [3, 0], [0, 0]], constant_values=0)
        x = tf.concat((x_shift_padded[:,:-3,:], x_pass), axis=-1)
        xvqk = self.convin(x)
        output = self.pos_enc(xvqk)
        attn_output = self.attn(
            query=output,
            key=output,
            values=xvqk,
        )

        acti = self.act(attn_output)
        acti = acti + 1
        output1 = xvqk * acti
        output = output1 + attn_output
        output = self.convin2(output)
        output = output + output1
        output = self.batchnrm(output)
        output = output + u
        return output


    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "n_head": self.n_head,
            "dropout": self.dropout_rate,
            "is_casual": self.is_casual,
        })
        return config