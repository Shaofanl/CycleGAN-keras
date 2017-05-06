from CycleGAN.utils.data_utils import ImageGenerator

if __name__ == '__main__':
    ig = ImageGenerator(root='./datasets/horse2zebra/trainA', 
                                    resize=(143,143), crop=(128,128))
    imgs = ig(10)
    print imgs.shape
