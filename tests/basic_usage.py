# this code can turn horse to zebra at around 20000 iteration
# when batch_size = 5 save_iter = 200 niter = 100000 lmbd = 10 pic_dir = args.pic_dir idloss = 0.0 lr = 0.00005
# can turn horse to zebra around 14000 iteration
# when batch_size = 1 save_iter = 200 niter = 100000 lmbd = 5 pic_dir = args.pic_dir idloss = 0.0 lr = 0.0002

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
    opt.batch_size = 1
    opt.save_iter = 200
    opt.niter = 100000
    opt.lmbd = 5 
    opt.pic_dir = args.pic_dir
    opt.idloss = 0.0
    opt.lr = 0.0002

    opt.__dict__.update(args.__dict__)
    opt.summary()


    cycleGAN = CycleGAN(opt)

    IG_A = ImageGenerator(root='./datasets/horse2zebra/trainA', 
                resize=opt.resize, crop=opt.crop)
    IG_B = ImageGenerator(root='./datasets/horse2zebra/trainB', 
                resize=opt.resize, crop=opt.crop)

    cycleGAN.fit(IG_A, IG_B)
