from CycleGAN.utils.data_utils import ImageGenerator
from CycleGAN.models import CycleGAN
from CycleGAN.utils import Option

if __name__ == '__main__':
    opt = Option()
    cycleGAN = CycleGAN(opt)

    IG_A = ImageGenerator(root='./datasets/horse2zebra/trainA', 
                resize=opt.resize, crop=opt.crop)
    IG_B = ImageGenerator(root='./datasets/horse2zebra/trainB', 
                resize=opt.resize, crop=opt.crop)

    cycleGAN.fit(IG_A, IG_B)
