from CycleGAN.models.gen.resnet import resnet_6blocks

if __name__ == '__main__':
    input_shape = (3, 224, 224)
    output_shape = (4, 224, 224)

    m = resnet_6blocks(input_shape=input_shape,
                       output_nc = output_shape[0],
                       ngf=64)
    assert m.output_shape[1:] == output_shape
