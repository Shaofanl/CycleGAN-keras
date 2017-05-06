from CycleGAN.models.gen.resnet import resnet_6blocks
from CycleGAN.models.dis.basic  import basic_D
from CycleGAN.models import CycleGAN
from CycleGAN.utils import Option

if __name__ == '__main__':
    input_shape = (3, 224, 224)
    output_shape = (4, 224, 224)

#   resnet_6blocks(input_shape=input_shape,
#                      output_nc = output_shape[0],
#                      ngf=64)
#   basic_D(input_shape=input_shape, ndf=64)

    CycleGAN(Option())

