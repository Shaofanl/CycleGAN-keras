# keras version of https://github.ckom/junyanz/CycleGAN/models/cycle_gan_model.lua

from .base import BaseModel
from .gen import defineG
from .dis import defineD
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

class CycleGAN(BaseModel):
    name = 'CycleGAN'
    def __init__(self, opt):
        gen_B = defineG(opt.which_model_netG, input_shape=opt.shapeA, output_shape=opt.shapeB, ngf=opt.ngf)
        dis_B = defineD(opt.which_model_netD, input_shape=opt.shapeB, ndf=opt.ndf, use_sigmoid=not opt.use_lsgan)

        gen_A = defineG(opt.which_model_netG, input_shape=opt.shapeB, output_shape=opt.shapeA, ngf=opt.ngf)
        dis_A = defineD(opt.which_model_netD, input_shape=opt.shapeA, ndf=opt.ndf, use_sigmoid=not opt.use_lsgan)


        # build for generators
        real_A = Input(opt.shapeA)
        fake_B = gen_B(real_A)
        dis_fake_B = dis_B(fake_B)

        real_B = Input(opt.shapeB)
        fake_A = gen_A(real_B)
        dis_fake_A = dis_A(fake_A)

        rec_A = gen_A(fake_B) # = gen_A(gen_B(real_A))
        rec_B = gen_B(fake_A) # = gen_B(gen_A(real_B))

        G_trainner = Model([real_A, real_B], 
                 [dis_fake_B,   dis_fake_A,     rec_A,      rec_B])
        
        G_trainner.compile(Adam(lr=opt.lr, beta_1=opt.beta1,),
            loss=['MSE',        'MSE',          'MAE',      'MAE'],
            loss_weights=[1,    1,              opt.lmbd,   opt.lmbd])
        # label:  0             0               real_A      real_B


        # build for discriminators 
        real_A = Input(opt.shapeA)
        fake_A = Input(opt.shapeA)
        real_B = Input(opt.shapeB)
        fake_B = Input(opt.shapeB)

        dis_real_A = dis_A(real_A)
        dis_fake_A = dis_A(fake_A)
        dis_real_B = dis_B(real_B)
        dis_fake_B = dis_B(fake_B)

        D_trainner = Model([real_A, fake_A, real_B, fake_B], 
                [dis_real_A, dis_fake_A, dis_real_B, dis_fake_B])
        D_trainner.compile(Adam(lr=opt.lr, beta_1=opt.beta1,), loss='MSE')
        # label: 0           0.9         0           0.9


        self.G_trainner = G_trainner
        self.D_trainner = D_trainner
        self.AtoB = gen_B
        self.BtoA = gen_A
        self.opt = opt

    def fit(self, img_A_generator, img_B_generator):
        opt = self.opt
        bs = opt.batch_size
        
        fake_A_pool = []
        fake_B_pool = []

        iteration = 0
        while iteration < opt.niter:
            # sample
            real_A = img_A_generator(bs)
            real_B = img_B_generator(bs)

            # fake pool
            fake_A_pool.append(self.BtoA.predict(real_B))
            fake_B_pool.append(self.AtoB.predict(real_A))
            fake_A_pool = fake_A_pool[:opt.pool_size] 
            fake_B_pool = fake_B_pool[:opt.pool_size]

            fake_A = [fake_A_pool[ind] for ind in np.random.choice(len(fake_A_pool), size=(bs,), replace=False)]
            fake_B = [fake_B_pool[ind] for ind in np.random.choice(len(fake_B_pool), size=(bs,), replace=False)]
            fake_A = np.array(fake_A)
            fake_B = np.array(fake_B)

            ones  = np.ones((bs,))
            zeros = np.zeros((bs,))

            # train
            _, G_loss_fake_B, G_loss_fake_A, G_loss_rec_A, G_loss_rec_B = \
                self.G_trainner.train_on_batch([real_A, real_B],
                    [zeros, zeros, real_A, real_B])

            _, D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B = \
                self.D_trainner.train_on_batch([real_A, fake_A, real_B, fake_B],
                    [zeros, ones*0.9, zeros, ones])

            print('Generator Loss:')
            print('fake_A: {:.3f} rec_A: {:.3f} | fake_B: {:.3f} rec_B: {:.3f}'.\
                    format(G_loss_fake_A, G_loss_rec_A, G_loss_fake_B, G_loss_rec_B))
            print('Discriminator Loss:')
            print('real_A: {:.3f} fake_A: {:.3f} | real_B: {:.3f} fake_B: {:.3f}'.\
                    format(D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B))
        
        
 
