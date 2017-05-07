import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, help="cuda", default='0')
parser.add_argument("--pic_dir", type=str, help="picture dir", default='./quickshots/')
args = parser.parse_args()
print args

import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',gpuarray.preallocate=0.00,device=cuda{}'.format(args.cuda)

from CycleGAN.utils.data_utils import ImageGenerator
from CycleGAN.models import CycleGAN
from CycleGAN.utils import Option

if __name__ == '__main__':
    opt = Option()
    opt.batch_size = 10
    opt.save_iter = 100
    opt.niter = 100000
    opt.lmbd = 5
    opt.pic_dir = args.pic_dir

    opt.__dict__.update(args.__dict__)


    cycleGAN = CycleGAN(opt)

    IG_A = ImageGenerator(root='./datasets/horse2zebra/trainA', 
                resize=opt.resize, crop=opt.crop)
    IG_B = ImageGenerator(root='./datasets/horse2zebra/trainB', 
                resize=opt.resize, crop=opt.crop)

    cycleGAN.fit(IG_A, IG_B)
