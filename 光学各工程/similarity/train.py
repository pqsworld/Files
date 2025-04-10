"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # opt.dataroot="/data/guest/zsn/simi_data_light/train1219"
    # opt.name='st_debug'
    # opt.model='pix2pix'
    # opt.direction='AtoB' 
    # opt.batch_size=512  
    # opt.phase=['0','1']
    # opt.load_size_h=72
    # opt.load_size_w = 72 
    # opt.input_nc = 2 
    # opt.output_nc =1 
    # opt.netG='bufen5_out2'
    # opt.init_type ='kaiming'
    # opt.ndf =4 
    # opt.ngf =4 
    # opt.gpu_ids=1,
    # opt.dataset_mode='simicsvlight'
    # opt.display_id=0
    # opt.out2=True
    # opt.teacher=False

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    loss_avg = OrderedDict()
    loss_avg_optimal = OrderedDict()
    epoch_optimal=1
    for name in model.loss_names:
            loss_avg_optimal[name]=100
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for name in model.loss_names:
            loss_avg[name]=0
        num=0
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            #if epoch_iter % opt.print_freq == 0:    # print training losses and save logging information to the disk
            if epoch_iter % (opt.batch_size*8) == 0: 
                losses = model.get_current_losses()
                #t_comp = (time.time() - iter_start_time) / opt.batch_size
                #print('loss:',losses)       
                for name in model.loss_names:
                    if name=='G_L1' or name=='G_L1_sm' or name=='G_L1_kong' :
                        loss_avg[name]+=losses[name]
                    else:
                        loss_avg[name]+=abs(losses[name]-0.693)			#ln(0.5)		
                num=num+1
                #visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            for name in model.loss_names:
                loss_avg[name]/=num
            print('epoch=%d,num=%d,'%(epoch,num),loss_avg)
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, loss_avg, t_comp, t_data)
            if loss_avg['G_L1'] < loss_avg_optimal['G_L1']:
                epoch_optimal = epoch
                for name in model.loss_names:
                    loss_avg_optimal[name]=loss_avg[name]
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
    print('epoch_optimal=%d,'%(epoch_optimal),loss_avg_optimal)
    visualizer.print_current_losses(epoch_optimal, epoch_iter, loss_avg_optimal, t_comp, t_data)