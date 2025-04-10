from PIL import Image
import torch
import sys
import torch.nn as nn


sys.path.append('..')
from .base_model import BaseModel
from . import networks


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='unaligned')
        # if is_train:
        #     #parser.set_defaults(pool_size=0, gan_mode='vanilla')
        #     parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1','G_L11'] #,'eva','G_L12','G_L112','ssim','stability','gr','TV'
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
     
        if self.opt.isTrain:
            self.visual_names = ['real_A', 'fake_B','real_B']#,'badmask','fake_B1','real_B1','fake_B2','real_B2','real_A1'] #, 'test_A', 'test_B'
        else:
            self.visual_names = ['real_A', 'fake_B','real_B']#,'badmask','fake_B1','real_B1','fake_B2','real_B2','real_A1'] #, 'test_A', 'test_B'

            # self.visual_names = ['real_A', 'fake_B', 'fake_BD','real_B'] #, 'test_A', 'test_B'
        # if self.opt.addtest:
        #     visuadd = ['test_A', 'test_B']
        #     self.visual_names.append(visuadd)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        self.opt = opt
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      True, opt.init_type, opt.init_gain, opt.alpha_w, self.gpu_ids)

        # if self.isTrain:
        #     self.netmask = networks.define_G(1, 1, 4, opt.netmask, opt.norm,
        #                               True, opt.init_type, opt.init_gain, opt.alpha_w, self.gpu_ids)
            # self.netmask.setup()
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.num_d, opt.n_layers_D,
                                          128, opt.norm, opt.init_type, opt.init_gain,
                                          self.gpu_ids)

        if self.isTrain:
            print_network(self.netG)
            # print_network(self.netD)
            # define loss functions
            if opt.netD == 'multiscale':
                self.criterionGAN = networks.GANLoss0(opt.gan_mode).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionL1 = torch.nn.SmoothL1Loss()#torch.nn.L1Loss()
    
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)
            self.maxpool = nn.MaxPool2d(2, 2)
            # model_weights="/home/zhangsn/enhance/enhance_newest/superGlue_50000_unsuperpoint_desc.pth.tar"
            # self.hardnet = load_model_hardnet_cls(model_weights, DO_CUDA=True)
        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # print(self.real_B)
        # self.blur_A = input['A_blur'].to(self.device)
            # self.blur_A = torch.where(self.real_C == 1, self.real_C, self.blur_A)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B = self.netG(self.real_A)#[-1].squeeze(0)
        # print(self.fake_B)
            # self.real_A = F.resize(self.real_A, [122,36])
            # print(self.fake_B.shape)
            # print(self.real_B.shape)
            # print(self.real_A.shape)
            # self.real_A2 = self.fake_B*0.2 + self.real_A*0.8
            # self.fake_B= self.netG(self.real_A2)
            # self.real_B_ez = torch.where(self.real_A > 0.8, self.fake_B, self.real_B)
            # self.fake_B[:,:,3:125,5:27] = self.real_B[:,:,3:125,5:27]#.to(self.device)
            # print(self.fake_B)
            # fake_B中心用real_B替代
            # self.fake_B = torch.where(self.real_A > 0.8, self.fake_B, self.real_B)
            # self.fake_B = torch.where(self.real_A < 100, self.fake_B, self.real_B)
            # # self.fake_B.data[:,:,3:125,5:27] = self.real_B.data[:,:,3:125,5:27]
            # self.fake_B[:,:,3:125,5:27] = self.real_B[:,:,3:125,5:27]
            # self.fake_B00 = self.fake_B * 0.1 + self.real_A * 0.9
            # self.fake_B = self.netG(self.fake_B00)
            # self.fake_B = self.netG(self.fake_B)
        # if self.opt.isTrain:
        #     self.badmask = self.netmask(self.real_A)
        #     self.badmask = torch.where(self.badmask < 0.5, -1.0, 1.0)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        # print(len(pred_fake))
        self.loss_D_fake = self.criterionGAN(pred_fake, False) #criterionGAN0
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) #criterionGAN0

        self.fakebb = self.maxpool(self.fake_B)
        self.realbb = self.maxpool(self.real_B)
        self.realaa = self.maxpool(self.real_A)
        fake_abb = torch.cat((self.realaa, self.fakebb),1)
        real_abb = torch.cat((self.realaa, self.realbb), 1)
        pred_fakeb = self.netD(fake_abb.detach())
        pred_realb = self.netD(real_abb.detach())

        self.loss_D_fake = self.loss_D_fake+self.criterionGAN(pred_fakeb, False)
        self.loss_D_real = self.loss_D_real+self.criterionGAN(pred_realb, True)
        # GAN loss (Fake Passability Loss)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 #+ self.loss_gr1 * 50
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = 0
        # Second, G(A) = B
        fake_B = self.fake_B
        real_B = self.real_B
        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1    
        # add multiscale loss
        self.fakebb = self.maxpool(fake_B)
        self.realbb = self.maxpool(real_B)
        self.loss_G_L11 = self.criterionL1(self.fakebb, self.realbb) * self.opt.lambda_L1
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L11
        # print(self.loss_G)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()  # set D's gradients to zero
        # self.backward_D()  # calculate gradients for D
        # self.optimizer_D.step()  # update D's weights

        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero      
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
