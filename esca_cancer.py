from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, merge, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import concatenate


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 16
    return get_output_length(width), get_output_length(height)


def conv_factory(x, nb_filter, weight_decay=0.):
    # Apply BatchNorm, Relu, 3x3 Conv2D, optional dropout
    x = BatchNormalization(mode=0, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=0.):
    # Apply BatchNorm, Relu, 1x1 Conv2D, optional dropout and MaxPooling2D
    x = BatchNormalization(mode=0, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_uniform', padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=0.):
    # Build a denseblock where the output of each conv_factory is fed to subsequent ones
    for i in range(nb_layers):
        concat_axis = 1 if K.image_dim_ordering() == "th" else -1
        x = conv_factory(x, growth_rate, weight_decay)
        x = concatenate([x, x], axis=concat_axis)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    return x

list_feat = []
if K.image_dim_ordering() == 'tf':
    concat_axis = 3
elif K.image_dim_ordering() == 'th':
    concat_axis = 1
for i in range(depth):
    x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
    list_feat.append(x)
    x = Concatenate(axis=concat_axis)(list_feat)
nb_filter = growth_rate * nb_dense_block
x = transition(x, nb_filter, dropout_rate, weight_decay)

def DenseNet(nb_classes, depth, nb_dense_block, growth_rate, nb_filter, input_tensor=None, input_shape=None, dropout_rate=None, weight_decay=1E-4, trainable=False):
    """Build the DenseNet model"""
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        input_shape = (input_shape, input_shape, 3)
    else:
        concat_axis = 1
        input_shape = (3, input_shape, input_shape)
    input = Input(shape=input_shape, tensor=input_tensor)
    x = __create_dense_net(nb_classes, input, True, nb_dense_block, growth_rate, dropout_rate, weight_decay, concat_axis)
    return x

if input_tensor is None:
    img_input = Input(shape=(None, None, 3))
else:
    if not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    model_input = img_input
    assert depth % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='initial_conv')(img_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate, weight_decay=weight_decay)

        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)

    # Final dense layer
    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5, name='conv_final_bn')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    densenet = Model(model_input, x, name='densenet')

def rpn(base_layers, num_anchors, filter_size):
    x = Conv2D(filter_size, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation=None, kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]




