import keras.backend as K

def get_filter_dim():
    '''
        Theano uses `channels_first`: (batch, channels, height, width)
        Tensorflow uses `channels_last`: (batch, height, width, channels) 
    '''
    data_format = K.image_data_format()
    if data_format == 'channels_first':
        return 1
    elif data_format == 'channels_last':
        return 3
    else:
        raise NotImplemented
