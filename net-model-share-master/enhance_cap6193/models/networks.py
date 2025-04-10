import functools
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from torch.optim import lr_scheduler




###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    # if norm_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    # elif norm_type == 'instance':
    #     norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # elif norm_type == 'none':
    #     def norm_layer(x): return Identity()
    # else:
    #     raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.9)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
            alpha_w=1, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    # print(netG) 
     
    if netG == 'resnet_416small2212':   # a32结构，描述子扩边   renset通道数32 分组卷积+可分离卷积更多
        net =  ResnetGenerator323_7_RSG_small2_212(input_nc,2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, num_dis, n_layers_D=3, height=128, norm='batch', init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GANLoss0(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss0, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                pred = pred.cuda()
                target_tensor = target_tensor.cuda()
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.se = semodule
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride,groups=4)

class ResnetGenerator323_7_RSG_small2_212(nn.Module):
    def __init__(self, intput_nc, res_blocks=1):
        super(ResnetGenerator323_7_RSG_small2_212, self).__init__()

        self.resblocks = res_blocks

        dkersize = 3
        de_nc = 4
        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(intput_nc, de_nc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(de_nc),
        )

        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(de_nc, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
       
        self.featrues_deconv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=dkersize, stride=2, padding=1),  # 16*16                        
            nn.BatchNorm2d(16),
        )
        

        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        
        
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
       

        self.featrues_resnet1 = nn.Sequential(  # 8*8
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )        
        
        self.featrues_upconv3 = nn.Sequential(             
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,groups=4),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16)
        )
        
        self.featrues_upconv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(8),
        )
        
        self.featrues_upconv1_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )      
        
    def forward(self, input): 
        # print(input.shape)
        # input = F.pad(input, pad=(2,2,1,1), mode='constant', value=1)  #122*36 124*40
        #融合版本原图输入,增强版本注释
        input = input[:,:,2:120,2:34]   #118*32
		
        input = F.pad(input, pad=(4,4,3,3), mode='constant', value=1)  
  
        # input = F.pad(input, pad=(4,4,2,2), mode='constant', value=1)  #124*36
        # print(input.shape)
        input1 = self.featrues_deconv1(input)   # 4 72 26
        # print(input1)
        input01 = self.featrues_deconv2_1(input1)
        input02 = self.featrues_deconv2_2(input01)
        input = torch.cat([input01, input02], 1)

        input = self.featrues_deconv3(input)

        # print(input)
        input11 = self.featrues_deconv4_1(input)
        # print(input11)
        input12 = self.featrues_deconv4_2(input11)
        # print(input12)
        input = torch.cat([input11, input12], 1)
        # print(input)
        out = input + self.featrues_resnet1(input)
        # print(out)
        out = self.featrues_upconv3(out)
        # print(out)
        # print(out.shape)
        out = self.featrues_upconv2(out)
        # print(out)
        out_u1 = self.featrues_upconv1_1(out)
        out_u2 = self.featrues_upconv1_2(out_u1)
        
        out = torch.cat([out_u1, out_u2], 1)  
        
        out = self.featrues_upconv1(out)   # 4 72 26
        # print(out)  

        # out = out*input1
        out = self.featrues_upconv0(out)
        # print(out.shape)
        # out[:,:,:,3:33]=input111[:,:,:,3:33]
        out = out[:,:,1:123,2:38]
        # out = out[:,:,1:123,:]   #122*36
        # out = out[:,:,2:182,4:40]  #top1 left2
        # print(out.shape)
        # import sys
        # sys.exit()        
        return out
class ResnetGenerator323_7_RSG_small2_212_expand(nn.Module):
    def __init__(self, intput_nc, res_blocks=1):
        super(ResnetGenerator323_7_RSG_small2_212_expand, self).__init__()

        self.resblocks = res_blocks

        dkersize = 3
        de_nc = 4
        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(intput_nc, de_nc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(de_nc),
        )

        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(de_nc, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
       
        self.featrues_deconv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=dkersize, stride=2, padding=1),  # 16*16                        
            nn.BatchNorm2d(16),
        )
        

        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        
        
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            # nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            DepthwiseSeparableConvolution(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
       

        self.featrues_resnet1 = nn.Sequential(  # 8*8
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )        
        
        self.featrues_upconv3 = nn.Sequential(             
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,groups=4),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16)
        )
        
        self.featrues_upconv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(8),
        )
        
        self.featrues_upconv1_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )      
        
    def forward(self, input): 
        # print(input.shape)
        # input = F.pad(input, pad=(2,2,1,1), mode='constant', value=1)  #122*36 124*40
        #融合版本原图输入,增强版本注释
        #input = input[:,:,2:120,2:34]   #118*32
		
        input = F.pad(input, pad=(8,8,3,3), mode='constant', value=1)  
  
        # input = F.pad(input, pad=(4,4,2,2), mode='constant', value=1)  #124*36
        # print(input.shape)
        input1 = self.featrues_deconv1(input)   # 4 72 26
        # print(input1)
        input01 = self.featrues_deconv2_1(input1)
        input02 = self.featrues_deconv2_2(input01)
        input = torch.cat([input01, input02], 1)

        input = self.featrues_deconv3(input)

        # print(input)
        input11 = self.featrues_deconv4_1(input)
        # print(input11)
        input12 = self.featrues_deconv4_2(input11)
        # print(input12)
        input = torch.cat([input11, input12], 1)
        # print(input)
        out = input + self.featrues_resnet1(input)
        # print(out)
        out = self.featrues_upconv3(out)
        # print(out)
        # print(out.shape)
        out = self.featrues_upconv2(out)
        # print(out)
        out_u1 = self.featrues_upconv1_1(out)
        out_u2 = self.featrues_upconv1_2(out_u1)
        
        out = torch.cat([out_u1, out_u2], 1)  
        
        out = self.featrues_upconv1(out)   # 4 72 26
        # print(out)  

        # out = out*input1
        out = self.featrues_upconv0(out)
        # print(out.shape)
        # out[:,:,:,3:33]=input111[:,:,:,3:33]
        # out = out[:,:,1:123,2:38]
        # out = out[:,:,1:123,:]   #122*36
        # out = out[:,:,2:182,4:40]  #top1 left2
        # print(out.shape)
        # import sys
        # sys.exit()        
        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class SEBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_in,ch_out,4,1,0,bias=False),
            hswish(),
            nn.Conv2d(ch_out,ch_out,1,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,fea_s,fea_b):
        # print(fea_s.shape,fea_b.shape)
        wei = self.block(fea_s)
        # print(wei.shape)
        return fea_b*wei
    

# 判别器
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()