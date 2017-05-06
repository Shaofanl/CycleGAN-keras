from .resnet import resnet_6blocks
from ...utils.backend_utils import get_filter_dim

def defineG(which_model_netG, input_shape, output_shape, ngf):
    output_nc = output_shape[get_filter_dim()-1]
    if which_model_netG == 'resnet_6blocks':
        return resnet_6blocks(input_shape, output_nc, ngf)
    else:
        raise NotImplemented

