from PIL import Image
import torch
import sys
import torch.nn as nn
import torchvision.transforms

sys.path.append('..')
from .base_model import BaseModel
from . import networks
from util import util
import numpy as np
import math
import scipy
import random

from .networks import ResnetGenerator323_7_RSG_un_2

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class ResnetTrans(nn.Module):
    def __init__(self):
        super(ResnetTrans, self).__init__()

        self.featrues_upconv = nn.Sequential(  # 8*8
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
    def forward(self, inputa,inputb):
        input = torch.cat((inputa, inputb), 1)
        out = self.featrues_upconv(input)
        # print(out[0,0,:,:])
        return out
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
        self.loss_names = ['G', 'G_L1','G_L11']#,'tea']
  
 
        if self.opt.isTrain:
            self.visual_names = ['real_A', 'fake_B','real_B']#,'teacher_B']#,'fake_Brh']
        else:
            # self.visual_names = ['real_A', 'fake_B','real_B']
            self.visual_names = ['enhance','enhance_1']#'merge','nlm'
            self.netRtoE = ResnetTrans()
            self.netRtoE.load_state_dict(torch.load("./pths/trans_net_G.pth",map_location=str(self.device)))
            self.netRtoE.to(str(self.device))
            self.netRtoE.eval()

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        self.opt = opt
        if self.isTrain:
            self.model_names = ['G']#,'RtoE']
            if self.opt.teacher:
                self.net_T = ResnetGenerator323_7_RSG_un_2(opt.input_nc, 2)
                self.net_T.load_state_dict(torch.load("./pths/ry58154_net_G.pth",map_location=str(self.device)))
                self.net_T.to(str(self.device))
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      True, opt.init_type, opt.init_gain,  self.gpu_ids)
        # self.netRtoE =networks.define_R(opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            print_network(self.netG)
            self.criterionL1 = torch.nn.SmoothL1Loss()#torch.nn.L1Loss()
    
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            
            self.maxpool = nn.MaxPool2d(2, 2)
            # self.optimizer_R = torch.optim.Adam(self.netRtoE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_R)
        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.dataset_mode == 'wealigned':
            self.mask_img = input['mask'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
  

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)#[-1].squeeze(0)
        # self.netG0 = self.netG.module
        # self.netG0.load_state_dict(torch.load("/home/zhangsn/enhance/02proj/checkpoints/netenlight/newGT_sm_241/latest_net_G.pth",map_location=str(self.device)))           
        # self.fake_Brh = self.netG0(self.real_A)
        # self.fake_B = self.netRtoE(self.real_A,self.fake_Brh) 
        if self.opt.teacher:
            self.teacher_B=self.net_T(self.real_A)
        if not self.isTrain:
            self.fake_B1=self.netRtoE(self.real_A,self.fake_B)
            self.enhance=self.fake_B
            self.enhance_1=self.fake_B1
            self.merge=self.fake_B
            self.nlm=self.fake_B1
                   
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

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 #+ self.loss_gr1 * 50
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator


        num_tmp5 = torch.tensor(0.5, dtype=torch.float32).cuda()
        num_tmp0 = torch.tensor(0.0, dtype=torch.float32).cuda()
        num_tmp1 = torch.tensor(1.0, dtype=torch.float32).cuda()

        # weight_tmp1 = torch.tensor(1, dtype=torch.float32).cuda()

        fab = torch.where(self.fake_B > num_tmp5, num_tmp0,num_tmp1)
        reb = torch.where(self.real_B > num_tmp5, num_tmp0,num_tmp1)

        hre0 = torch.tensor(0, dtype=torch.float32).cuda()
        hre1 = torch.tensor(1, dtype=torch.float32).cuda()
        B_f = torch.where(self.mask_img == -1, self.fake_B, hre0)
        B_r = torch.where(self.mask_img == -1, self.real_B, hre0)
        bw_f = torch.where(self.mask_img==-1,fab,hre0)
        bw_r = torch.where(self.mask_img==-1,reb,hre0)

        b2_f = torch.where(self.mask_img == -1, self.fake_B, hre0)
        b2_r = torch.where(self.mask_img == -1, self.real_B, hre0)
        self.fakebb = self.maxpool(b2_f)
        self.realbb = self.maxpool(b2_r)

        self.loss_G_L1 = self.criterionL1(B_f,B_r) * self.opt.lambda_L1 +self.criterionL1(bw_f,bw_r)*5 
        self.loss_G_L11 = self.criterionL1(self.fakebb, self.realbb) * self.opt.lambda_L1
        
        # junheng mianji
        b,c,h,w=self.mask_img.shape
        all = b*c*h*w
        tempmask = self.mask_img.view(1,all)
        num = len(torch.where(tempmask == -1)[0])
        # tempmask0 = self.mask_img0.view(1,all)
        # num0 = len(torch.where(tempmask0 == -1)[0])
        # print(num)
        self.loss_G_L1 = self.loss_G_L1*num/all
        self.loss_G_L11 = self.loss_G_L11*num/all
        
        self.loss_G = self.loss_G_L1 + self.loss_G_L11


        if self.opt.teacher:
            self.loss_tea = self.criterionL1(self.fake_B,self.teacher_B) * self.opt.lambda_L1
            alpha=0.5   # 原 loss调整占比
            self.loss_G = self.loss_G*alpha + self.loss_tea*(2-alpha)
        self.loss_G.backward(retain_graph=True)
   
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
        
        # self.set_requires_grad(self.netG, False)
        # self.optimizer_R.zero_grad()  # set G's gradients to zero    
        # self.backward_G()  # calculate graidents for G      
        # self.optimizer_R.step()  # udpate G's weights
