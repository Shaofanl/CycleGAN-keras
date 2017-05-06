#############################################################
# We use zeropadding to replace all SpatialReflectionPadding
#   by claiming `padding='same'`

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Input
from keras.layers.merge import Concatenate
from keras.models import Model

from ...utils.backend_utils import get_filter_dim


def res_block(input, filters, kernel_size=(3,3), strides=(1,1), padding='same'):
    x = Conv2D(filters=filters, 
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,)(input)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, 
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,)(x)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    merged = Concatenate(axis=get_filter_dim())([input, x])
    return merged

def resnet_6blocks(input_shape, output_nc, ngf, **kwargs):
    ks = 3
    f = 7
    p = (f-1)/2

    input = Input(input_shape)
    x = Conv2D(ngf, (f,f), padding='same')(input)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    x = Conv2D(ngf*2, (ks,ks), strides=(2,2), padding='same')(x)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    x = Conv2D(ngf*4, (ks,ks), strides=(2,2), padding='same')(x)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)
    x = res_block(x, ngf*4)

    x = Conv2DTranspose(ngf*2, (ks,ks), strides=(2,2), padding='same')(x)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)
    
    x = Conv2DTranspose(ngf, (ks,ks), strides=(2,2), padding='same')(x)
    x = BatchNormalization(axis=get_filter_dim())(x)
    x = Activation('relu')(x)

    x = Conv2D(output_nc, (f,f), padding='same')(x)
    x = Activation('tanh')(x)
    
    model = Model(input, x, name=kwargs.get('name',None))
    print('Model resnet 6blocks:')
    model.summary()
    return model



