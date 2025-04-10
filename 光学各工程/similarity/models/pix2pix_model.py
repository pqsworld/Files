import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import numpy as np
from .MobileNet import *
import random
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_L1_sm','G_L1_kong','G_reg']
        if self.opt.teacher:
            self.loss_names.append('G_tea')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['classify','gt','gt_c', 'real_B' ]#['real_A','fake_B', 'real_B'] 'real_A','fake_B',
        #self.visual_names = ['real_A','fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']#, 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.block_nums)
        model_parameters = filter(lambda p: p.requires_grad, self.netG.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        if self.opt.teacher:
            self.netG_T = MNV3_bufen_new5_bin_reg(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=1).cuda()
            # self.pthpath_T = "/hdd/file-input/zhangsn/light_simi/checkpoints/st_simi_1115_out2_sz96_frradd5/288_net_G.pth"
            self.pthpath_T = "/hdd/file-input/zhangsn/light_simi/checkpoints/st_simi_1205_out2_sz96_fr5_fa5_moredata_crop/217_net_G.pth"
            self.netG_T.load_state_dict(torch.load(self.pthpath_T,map_location='cuda'))
            # print(params)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # self.criterion = nn.CrossEntropyLoss(torch.Tensor([3,1])).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss()  #class [0,c]
            #self.criterionL1 = nn.BCELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.gt = input['gt'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.clcgt = input['flag'].to(self.device)#.to(torch.float32)
        if self.opt.teacher:
            self.real_A_tea = input['A_tea'].to(self.device)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.cropflag:
            h,w=self.opt.load_size_h, self.opt.load_size_w
            scale=int(h/96*10)
            if random.choice([0,1,2,3])==0:
            # if random.choice([0,1,2])==0:
                left=random.randint(0,scale)
                right = w-random.randint(0,scale)
                bottom = random.randint(0,scale)
                top = h-random.randint(0,scale)
                self.real_A = self.real_A[:,:,left:right,bottom:top]
                # if self.opt.teacher:
                #     left=random.randint(0,10)
                #     right = w-random.randint(0,10)
                #     bottom = random.randint(0,10)
                #     top = h-random.randint(0,10)
                #     self.real_A_tea = self.real_A_tea[:,:,left:right,bottom:top]
                # print(self.real_A.shape)
        if self.opt.out2:
            self.classify,self.clc_res = self.netG(self.real_A)  # G(A)
                
        else:
            self.classify = self.netG(self.real_A)  # G(A)
        if self.opt.teacher:
            self.classify_t,self.clc_res_t = self.netG_T(self.real_A_tea)
        self.loss_G_GAN = 0
        self.loss_G_L1 = 0
        self.loss_G_L1_kong = 0
        
        self.loss_G_L1_sm = 0
        self.loss_G_L1_kong = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0
        self.fake_B = self.real_B
        self.gt = self.gt.view(-1,1)  #B,->B,1
        self.gt_c = self.gt  

        self.clcgt = self.clcgt.view(-1,1)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #print(self.real_A.shape)
        #print(self.fake_B.shape)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self,epoch):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = 0
        self.loss_G_L1 = 0
        self.loss_G_L1_kong = 0
        
        self.loss_G_L1_sm = 0
        self.loss_G_L1_kong = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0

        
        #debug here
        # print(self.classify.shape)
        # print(self.classify[0,0], self.gt[0,0])
        self.loss_G_L1_sm = self.criterionL1(self.classify[:,0], self.gt[:,0])*100
        # print(self.loss_G_L1_sm)

        if self.opt.out2:
            self.loss_G_reg=self.criterion(self.clc_res, self.clcgt[:,0])*150#200#100  #分类头性能  [512,cls_num] [512]
            # self.loss_G_reg=self.criterionL1(self.regression[:,0], self.clcgt[:,0])*50
        else:
            self.loss_G_reg = 0
        

        self.loss_G = self.loss_G_L1_sm+self.loss_G_reg# + self.loss_G_L1_kong

        if self.opt.teacher:
            # self.loss_G_tea=self.criterionL1(self.classify[:,0], self.classify_t[:,0])*100+self.criterion(self.clc_res, self.clcgt[:,0])*150
            self.loss_G_tea=self.criterionL1(self.classify[:,0], self.classify_t[:,0])*50+self.criterion(self.clc_res, self.clcgt[:,0])*50
            self.loss_G = self.loss_G+self.loss_G_tea
        #self.loss_G = self.loss_G_L1  + self.loss_G_L1_sm #+ self.loss_G_L1_kong #+ self.loss_G_GAN
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #self.loss_G = self.loss_G_L1_sm
        self.loss_G.backward()

    def optimize_parameters(self,epoch):
    
        self.forward()  # compute fake images: G(A)
        # update D
        #if(epoch%5 == 0):
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.optimizer_D.zero_grad()  # set D's gradients to zero
        #self.backward_D()  # calculate gradients for D
        
        #self.optimizer_D.step()  # update D's weights
        # update G
        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G(epoch)  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
