import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, UpSampling3D, Subtract
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Dense, Add, Reshape, Permute, Flatten
from tensorflow.keras.layers import Concatenate, TimeDistributed, Lambda
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import BatchNormalization, Concatenate, ELU
from tensorflow.keras.regularizers import l2

from layers import InstanceNormalization, SplitChannels, BatchStd
from layers import SNDense, SNConv2D, SNConv3D
from layers import ReflectionPadding2D, ReflectionPadding3D


def dense_block(channels, norm=None, activation='leakyrelu'):

    D = SNDense if norm=='spectral' else Dense

    def block(x):
        
        if norm=="instance":
            x = InstanceNormalization(scale=False)(x)
        elif norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        if activation == 'leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'relu':
            x = ReLU()(x)
        elif activation == 'elu':
            x = ELU()(x)
        x = D(channels)(x)
        return x

    return block


def conv_block(channels, conv_size=(3,3),
    norm=None, stride=1, activation='leakyrelu', padding='reflect'):

    Conv = SNConv2D if norm=='spectral' else Conv2D

    def block(x):
        if norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        elif norm=="instance":
            x = InstanceNormalization(scale=False, axis=-1)(x)
        if activation == 'leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'relu':
            x = ReLU()(x)
        elif activation == 'elu':
            x = ELU()(x)
        if padding == 'reflect':
            pad = [(s-1)//2 for s in conv_size]
            x = ReflectionPadding2D(padding=pad)(x)
        x = Conv(channels, conv_size, 
            padding='valid' if padding=='reflect' else padding,
            strides=(stride,stride))(x)
        return x

    return block


def res_block(channels, conv_size=(3,3), stride=1, upscale=1, norm=None,
    activation="leakyrelu"):

    def block(x):
        in_channels = int(x.shape[-1])
        x_in = x
        if (stride > 1):
            x_in = AveragePooling2D(pool_size=(stride,stride))(x_in)
        elif (upscale > 1):
            x_in = UpSampling2D(upscale)(x_in)
            x = x_in

        if channels != in_channels:
            x_in = conv_block(channels, conv_size=(1,1), stride=1, 
                activation=False)(x_in)

        x = conv_block(channels, conv_size=conv_size, stride=stride,
            padding='same', norm=norm, activation=activation)(x)
        x = conv_block(channels, conv_size=conv_size, stride=1,
            padding='same', norm=norm, activation=activation)(x)

        x = Add()([x,x_in])

        return x

    return block


def conv_block_3d(channels, conv_size=(3,3,3),
    norm=None, stride=1, activation=True, padding='reflect'):

    Conv = SNConv3D if norm=='spectral' else Conv3D

    def block(x):
        if norm=="batch":
            x = BatchNormalization(momentum=0.8, scale=False)(x)
        elif norm=="instance":
            x = InstanceNormalization(scale=False, axis=-1)(x)
        if activation == 'leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'relu':
            x = ReLU()(x)
        if padding == 'reflect':
            pad = [(s-1)//2 for s in conv_size]
            x = ReflectionPadding3D(padding=pad)(x)
        x = Conv(channels, conv_size,
            padding='valid' if padding=='reflect' else padding,
            strides=(stride,stride,stride))(x)
        return x

    return block


def res_block_3d(channels, conv_size=(3,3,3), stride=1, norm=None,
    activation="leakyrelu"):

    def block(x):
        in_channels = int(x.shape[-1])
        x_in = x
        if (stride > 1):
            x_in = AveragePooling3D(pool_size=(stride,stride,stride))(x_in)

        if channels != in_channels:
            x_in = conv_block_3d(channels, conv_size=(1,1,1), stride=1, 
                activation=False)(x_in)

        x = conv_block_3d(channels, conv_size=conv_size, stride=stride,
            padding='same', norm=norm, activation=activation)(x)
        x = conv_block_3d(channels, conv_size=conv_size, stride=1,
            padding='same', norm=norm, activation=activation)(x)

        x = Add()([x,x_in])

        return x

    return block


def downsample_model(img_shape, num_scalings, activation='leakyrelu',
    norm=None):

    img_in_single = Input(shape=img_shape+(1,))
    x = img_in_single
    for i in range(num_scalings):
        num_channels = min(32 * 2**i, 64)
        block_norm = None if (i==0 and norm=='instance') else norm
        x = res_block(num_channels, stride=2,
            activation=activation, norm=block_norm)(x)
    x = Flatten()(x)
    feat_out_single = dense_block(1024,norm=norm)(x)
    return Model(inputs=img_in_single, outputs=feat_out_single)


def generator(proj_shape=(128,128), grid3d_shape=(32,32,32), noise_dim=64,
    num_scalings=3, num_dense=4, num_hidden=2, num_proj=3):

    proj_in = Input(shape=proj_shape+(num_proj,))
    noise_in = Input(shape=(noise_dim,))

    num_scalings_proj = int(round(np.log2(proj_shape[0]//4)))
    num_scalings_3d = int(round(np.log2(grid3d_shape[0]//4)))

    downsample = downsample_model(proj_shape, num_scalings_proj,
        activation='relu', norm='instance')

    x = SplitChannels()(proj_in)
    x = Concatenate()([downsample(xx) for xx in x])

    for i in range(num_dense//2):
        norm = None if i==0 else 'instance'
        x = dense_block(1024, activation='relu', norm=norm)(x)

    n = Dense(1024)(noise_in)
    x = Multiply()([x,n])

    for i in range(num_dense-num_dense//2):
        norm = None if i==0 else 'instance'
        activation = None if i==0 else 'relu'
        x = dense_block(1024, activation=activation, norm=norm)(x)

    init_3d_shape = tuple(s//2**num_scalings for s in grid3d_shape) + (32,)
    x = dense_block(np.prod(init_3d_shape), activation='relu',
        norm='instance')(x)
    x = Reshape(init_3d_shape)(x)

    for i in range(num_hidden):
        x = res_block_3d(256, activation='relu', norm='instance')(x)

    for i in range(num_scalings_3d):
        num_channels = 128 // 2**i
        x = UpSampling3D(2)(x)
        x = conv_block_3d(num_channels, activation='relu',
            norm='instance')(x)

    grid3d_out = conv_block_3d(1, conv_size=(3,3,3), padding='same',
        activation='relu', norm='instance')(x)
    
    gen = Model(inputs=[proj_in, noise_in],
        outputs=grid3d_out)

    return gen


def discriminator(proj_shape=(128,128), grid3d_shape=(32,32,32),
    num_hidden=2, num_proj=3):

    proj_in = Input(shape=proj_shape+(num_proj,))
    grid3d_in = Input(shape=grid3d_shape+(1,))

    num_scalings_proj = int(round(np.log2(proj_shape[0]//4)))
    num_scalings_3d = int(round(np.log2(grid3d_shape[0]//4)))

    downsample = downsample_model(proj_shape, num_scalings_proj,
        norm='spectral')

    x = SplitChannels()(proj_in)
    x = Concatenate()([downsample(xx) for xx in x])

    x_3d = grid3d_in
    for i in range(num_scalings_3d):
        num_channels = 32 * 2**i
        x_3d = res_block_3d(num_channels, stride=2,
            norm='spectral')(x_3d)

    x_3d = conv_block_3d(256, conv_size=(4,4,4), 
        padding='valid', norm='spectral')(x_3d)
    x_3d = Flatten()(x_3d)

    #x = Concatenate()([x,x_3d])
    #for i in range(num_hidden):
    #    x = dense_block(1024, norm='spectral')(x)
    #x = LeakyReLU(0.2)(x)

    for i in range(num_hidden):
        x = dense_block(1024, norm='spectral')(x)
    for i in range(num_hidden):
        x_3d = dense_block(1024, norm='spectral')(x_3d)
    x = Multiply()([x,x_3d])

    x = dense_block(1024, activation=None, norm='spectral')(x)
    x = dense_block(1024, norm='spectral')(x)
    x = Concatenate()([x, BatchStd()(x)])
    x = dense_block(512, norm='spectral')(x)

    disc_out = dense_block(1, norm='spectral')(x)

    disc = Model(
        inputs=[proj_in, grid3d_in],
        outputs=disc_out
    )

    return disc


def predictor(proj_shape=(128,128), grid3d_shape=(32,32,32),
    num_hidden=2, num_proj=3):

    proj_in = Input(shape=proj_shape+(num_proj,))
    log_proj_size_in = Input(shape=(1,))

    num_scalings_proj = int(round(np.log2(proj_shape[0]//4)))

    downsample = downsample_model(proj_shape, num_scalings_proj,
        norm='batch')

    x = SplitChannels()(proj_in)
    x = Concatenate()([downsample(xx) for xx in x]+[log_proj_size_in])

    for i in range(num_hidden):
        x = dense_block(1024, norm='batch')(x)

    x = dense_block(512, norm='batch')(x)
    x = dense_block(512, norm='batch')(x)
    x = dense_block(512, norm='batch')(x)

    log_mass_out = Dense(1, name='log_mass')(x)
    log_grid3d_size_out = Dense(1, name='log_grid3d_size')(x)

    pred = Model(
        inputs=[proj_in, log_proj_size_in],
        outputs=[log_mass_out, log_grid3d_size_out]
    )

    pred.compile(
        loss=['mse', 'mse'],
        optimizer='adam'
    )

    return pred



def triplet_validator(proj_shape=(128,128), grid3d_shape=(32,32,32),
    num_hidden=2, num_proj=3):

    proj_in = Input(shape=proj_shape+(num_proj,))

    num_scalings_proj = int(round(np.log2(proj_shape[0]//4)))

    downsample = downsample_model(proj_shape, num_scalings_proj,
        norm='batch')

    x = SplitChannels()(proj_in)
    x = Concatenate()([downsample(xx) for xx in x])

    for i in range(num_hidden):
        x = dense_block(1024, norm='batch')(x)

    x = dense_block(512, norm='batch')(x)
    x = dense_block(512, norm='batch')(x)
    x = dense_block(512, norm='batch')(x)

    valid_triplet = Dense(1, activation='sigmoid',
        name='valid_triplet')(x)

    pred = Model(
        inputs=proj_in,
        outputs=valid_triplet
    )

    pred.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return pred
