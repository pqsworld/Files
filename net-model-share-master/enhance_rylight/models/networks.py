import functools
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
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
             gpu_ids=[]):
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
    
    if netG == 'resnet_enhance':
        net = ResnetGenerator323_7_RSG_un_2(input_nc, 2)  
    elif netG == 'resnet_rynew241':     #最终结构
        net = ResnetGenerator_rynew_2_41(input_nc, 2)
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
        return out


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size,padding=(kernel_size//2), bias=bias, stride = 1,groups=4)

    def forward(self, x, x_1):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(x1)
        # print(x2[0,0,:,:])
        # exit()
        out=x*x2+x_1*(1-x2)
        return out
        
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

class ResnetGenerator323_7_RSG_un_2(nn.Module):
    def __init__(self, intput_nc, res_blocks=1):
        super(ResnetGenerator323_7_RSG_un_2, self).__init__()

        self.resblocks = res_blocks

        dkersize = 3

        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(intput_nc, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
        )

        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=dkersize, stride=1, padding=1),  # 64
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=dkersize, stride=1, padding=2, dilation=2),  # 32*32
            nn.BatchNorm2d(4),
        )

        self.featrues_deconv2_3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=dkersize, stride=1, padding=3, dilation=3),  # 32*32
            nn.BatchNorm2d(4),
        )

        self.featrues_deconv2_4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=dkersize, stride=1, padding=4, dilation=4),  # 32*32
            nn.BatchNorm2d(4),
        )
        # ******
        self.featrues_deconv3 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=dkersize, stride=2, padding=1),  # 32
            nn.BatchNorm2d(32),
        )

        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(16),
        )
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=dkersize, stride=1, padding=2, dilation=2),  # 32*32
            nn.BatchNorm2d(16),
        )
        # ******
        self.featrues_deconv5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 8*8
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=1, stride=1, padding=0),  # 8*8
            # nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )

        self.featrues_resnet1 = nn.Sequential(  # 32

            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        self.featrues_resd1 = nn.Sequential(  # 16
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(96),
            # nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=4),  # 20736

            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.featrues_resd2 = nn.Sequential(  # 8
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(96),
            # nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=4),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.featrues_resnet2 = nn.Sequential(  # 8*8

            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        self.featrues_resnet3 = nn.Sequential(  # 8*8

            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        self.featrues_fc = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.featrues_upconv5 = nn.Sequential(
            # nn.ReLU(True),
            # nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(24),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=4),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48)
        )
        self.featrues_upconv4 = nn.Sequential(
            # nn.ReLU(True),
            # nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1,groups=4),#20736
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(48),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=4),  # 20736
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48)
        )
        # *****
        self.featrues_upconv4_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        # ****
        self.featrues_upconv4_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(16),
        )
        # ****
        self.featrues_upconv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
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
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=2, dilation=2),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=3, dilation=3),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=4, dilation=4),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(8),

        )
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )
        self.sam1 = SAM(48, 1, False)
        self.sam2 = SAM(48, 1, False)

    def forward(self, input):
        # print(input.shape)
        input1 = self.featrues_deconv1(input)
        # print(input[0,0,0,:])
        # exit()
        input01 = self.featrues_deconv2_1(input1)

        # print(input01[0,0,:,:])
        # exit()
        input02 = self.featrues_deconv2_2(input1)

        input03 = self.featrues_deconv2_3(input1)

        input04 = self.featrues_deconv2_4(input1)
        input = torch.cat([input01, input02, input03, input04], 1)
        # print(input02[0,0,:,:])
        # exit()
        input = self.featrues_deconv3(input)
        # print(input[0,0,:,:])
        # exit()
        input11 = self.featrues_deconv4_1(input)
        input12 = self.featrues_deconv4_2(input)
        input = torch.cat([input11, input12], 1)
        input = self.featrues_deconv5(input)

        # print(input)
        # exit()
        out = input + self.featrues_resnet1(input)
        # exit()
        out_d3 = self.featrues_resd1(out)
        # print(out_d3[0,0,:,:])
        # exit()
        out_d3 = out_d3 + self.featrues_resnet2(out_d3)
        out_d4 = self.featrues_resd2(out_d3)
        out_d4 = out_d4 + self.featrues_resnet3(out_d4)

        # print("outd4",out_d4.shape)
        out_r = torch.mean(out_d4, [0, 1])
        # print("outr",out_r.shape)
        tt = torch.ones(out_d4.shape).cuda()


        out_r = torch.unsqueeze(out_r, 0)
        out_r = torch.unsqueeze(out_r, 0)
        # print("outr",out_r.shape)
        out111 = self.featrues_fc(out_r)
        # print("out111",out111.shape)
        b, c, w, h = out_d3.shape
        # print("out_d3",out_d3.shape)
        for i in range(b):
            tt[i, :, :, :] = out111[0, :, :, :]
        # print(tt.shape)
        out_d4 = tt + out_d4

        # exit()
        out_d4 = self.featrues_upconv5(out_d4)
        # print(out_d4[0,0,:,:])
        # exit()
        out_d4 = self.sam1(out_d3, out_d4)
        # print(out_d3[0,0,:,:])
        # exit()
        out_d3 = self.featrues_upconv4(out_d4)
        # print(out.size(),out_d3.size())
        # exit()
        out_d3 = self.sam2(out, out_d3)
        # out=out*2
        # print(out[0,0,:,:])
        # exit()
        # print(out[0,0,:,:])
        # exit()
        out0 = self.featrues_upconv4_1(out_d3)
        out1 = self.featrues_upconv4_2(out_d3)
        # print(out0)
        # exit()
        # out2=self.featrues_upconv4_3(out)
        out = torch.cat([out0, out1], 1)
        out = self.featrues_upconv3(out)
        out = self.featrues_upconv2(out)
        # print(out)
        # exit()
        out_u1 = self.featrues_upconv1_1(out)
        out_u2 = self.featrues_upconv1_2(out)
        out_u3 = self.featrues_upconv1_3(out)
        out_u4 = self.featrues_upconv1_4(out)
        out = torch.cat([out_u1, out_u2, out_u3, out_u4], 1)
        out = self.featrues_upconv1(out)
        out = out*input1
        out = self.featrues_upconv0(out)
        # print(out)
        # exit()
        return out

class ResnetGenerator_rynew_2_41(nn.Module):
    def __init__(self, intput_nc, res_blocks=1):
        super(ResnetGenerator_rynew_2_41, self).__init__()

        self.resblocks = res_blocks

        dkersize = 3

        self.featrues_deconv1 = nn.Sequential(  # 64*64
            # nn.ReflectionPad2d(1),
            nn.Conv2d(intput_nc, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
        )

        self.featrues_deconv2_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=dkersize, stride=1, padding=1),  # 64
            nn.BatchNorm2d(4),
        )
        self.featrues_deconv2_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )

        self.featrues_deconv2_3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )

        self.featrues_deconv2_4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(4),
        )
        # ******
        self.featrues_deconv3 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=dkersize, stride=2, padding=1),  # 32
            nn.BatchNorm2d(32),
        )

        self.featrues_deconv4_1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(16),
        )
        self.featrues_deconv4_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=dkersize, stride=1, padding=1),  # 32*32
            nn.BatchNorm2d(16),
        )
        # ******
        self.featrues_deconv5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConvolution(32, 64, kernel_size=3, stride=1, padding=1),  # 8*8
            # nn.BatchNorm2d(32),
            # nn.Conv2d(32, 96, kernel_size=1, stride=1, padding=0),  # 8*8
            # nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )

        self.featrues_resnet1 = nn.Sequential(  # 32

            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        self.featrues_resd1 = nn.Sequential(  # 16
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(96),
            # nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=12),  # 20736
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.featrues_resd2 = nn.Sequential(  # 8
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(96),
            # nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=12),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.featrues_resnet2 = nn.Sequential(  # 8*8
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        self.featrues_resnet3 = nn.Sequential(  # 8*8

            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            DepthwiseSeparableConvolution(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(48),
        )
        # self.featrues_fc = nn.Sequential(
        #     nn.Conv2d(1, 48, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(48),
        #     nn.LeakyReLU(0.2),
        # )
        self.featrues_upconv5 = nn.Sequential(
            # nn.ReLU(True),
            # nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(24),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=12),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48)
        )
        self.featrues_upconv4 = nn.Sequential(
            # nn.ReLU(True),
            # nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1,groups=4),#20736
            # nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(48),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=12),  # 20736
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48)
        )
        # *****
        self.featrues_upconv4_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        # ****
        self.featrues_upconv4_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
        )
        # ****
        self.featrues_upconv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,groups=4),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,groups=4),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
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
        self.featrues_upconv1_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(4),
        )
        self.featrues_upconv1 = nn.Sequential(
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,groups=4),  # 32*32  featrues_upconv3
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),  # 32*32  featrues_upconv3
            nn.BatchNorm2d(8),
        )
        self.featrues_upconv0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 32*32  featrues_upconv3
            # nn.BatchNorm2d(1),
            nn.Tanh()
        )
        self.sam1 = SAM(48, 1, False)
        self.sam2 = SAM(48, 1, False)

    def forward(self, input):
        # print(input.shape)
        # #融合版本原图输入
        # input = F.pad(input, pad=(2,2,2,2), mode='constant', value=-1)  
        # print(input.shape)
        input1 = self.featrues_deconv1(input)  #64*64*8(2)
        # print(input[0,0,0,:])
        # exit()
        input01 = self.featrues_deconv2_1(input1)  #64*64*4

        # print(input01[0,0,:,:])
        # exit()
        input02 = self.featrues_deconv2_2(input01)  #64*64*4

        input03 = self.featrues_deconv2_3(input02)  #64*64*4

        input04 = self.featrues_deconv2_4(input03)  #64*64*4
        input = torch.cat([input01, input02, input03, input04], 1)  #64*64*16(4)
        # print(input02[0,0,:,:])
        # exit()
        input = self.featrues_deconv3(input)  #32*32*32(2)
        # print(input[0,0,:,:])
        # exit()
        input11 = self.featrues_deconv4_1(input)  #32*32*16(1)
        input12 = self.featrues_deconv4_2(input11)  #32*32*16(1)
        input = torch.cat([input11, input12], 1)  #32*32*32(2)
        input = self.featrues_deconv5(input)  #32*32*64 32*32*48(4 3)

        # print(input)
        # exit()
        out = input + self.featrues_resnet1(input)  #32*32*48(3)
        # exit()
        out_d3 = self.featrues_resd1(out)   #16*16*48(0.75)
        # print(out_d3[0,0,:,:])
        # exit()
        out_d3 = out_d3 + self.featrues_resnet2(out_d3)   #16*16*48(0.75)
        out_d4 = self.featrues_resd2(out_d3)   #8*8*48(0.375)
        out_d4 = out_d4 + self.featrues_resnet3(out_d4)  #8*8*48(0.375)

        out_d4 = self.featrues_upconv5(out_d4)  #16*16*48(0.75)
        # print(out_d4[0,0,:,:])
        # exit()
        out_d4 = self.sam1(out_d3, out_d4)
        # print(out_d3[0,0,:,:])
        # exit()
        out_d3 = self.featrues_upconv4(out_d4)   #32*32*48(3)
        # print(out.size(),out_d3.size())
        # exit()
        out_d3 = self.sam2(out, out_d3)
        # out=out*2
        # print(out[0,0,:,:])
        # exit()
        # print(out[0,0,:,:])
        # exit()
        out0 = self.featrues_upconv4_1(out_d3)   #32*32*16
        out1 = self.featrues_upconv4_2(out0)  #32*32*16
        # print(out0)
        # exit()
        # out2=self.featrues_upconv4_3(out)
        out = torch.cat([out0, out1], 1)  #32*32*32(2)
        out = self.featrues_upconv3(out)   #64*64*16(4)
        out = self.featrues_upconv2(out)  #64*64*8 (2)
        # print(out)
        # exit()
        out_u1 = self.featrues_upconv1_1(out)  #64*64*4
        out_u2 = self.featrues_upconv1_2(out_u1)  #64*64*4
        out_u3 = self.featrues_upconv1_3(out_u2)  #64*64*4
        out_u4 = self.featrues_upconv1_4(out_u3)  #64*64*4
        out = torch.cat([out_u1, out_u2, out_u3, out_u4], 1)  #64*64*16 (4)
        out = self.featrues_upconv1(out)   #64*64*8(2)
        out = out*input1
        out = self.featrues_upconv0(out)  #128*128*1
        # out = out[:,:,2:126,2:126]
        # print(out)
        # exit()
        return out

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



#TRANS
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
