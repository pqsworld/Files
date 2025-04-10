"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
# from util.visualizer import make_dataset
from util import html
import pandas as pd
import matplotlib.pyplot as plt
#from mask2img import addmask2img
#from Dice import DiceJudge,AreaJudge
# from pathlib import Path


def _test(opt,model):
    # r1diff = []
    # r0diff = []
    # r1,r0 = 0,0
    # alldiff = []
    imglist = []
    gtlist = []
    reslist = []
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    allttest=len(dataset)
    for i, data in enumerate(dataset):
        # if i >= opt.num_test*test_frequency:  # only apply our model to opt.num_test images.
        #     break
        # if i % test_frequency != 0:
        #     continue
        model.set_input(data)  # unpack data from data loader      
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 50 == 0: 
            print('%s******%d/%d: %s' % (opt.phase,i, allttest,img_path))
        
        classify_c = visuals['classify']
        gt_q = visuals['gt']
        # res=(classify_c[0][0].numpy()*260+55).astype(int)
        # gt =(gt_q[0][0].numpy()*260+55).astype(int)
        res=(classify_c[0][0].numpy()*128+127).astype(int)
        gt =(gt_q[0][0].numpy()*128+127).astype(int)

        # difftt = abs(res-gt)
        # alldiff.append(difftt)
        imglist.append(img_path)
        reslist.append(res)
        gtlist.append(gt)
    
        # rec = img_path[0].split('_')[-4]
        # if rec == 'r1':
        #     r1diff.append(difftt)
        #     if difftt>10:
        #         r1+=1
        # else:
        #     r0diff.append(difftt)
        #     if difftt>10:
        #         r0+=1
   
    # print('r1>10',r1,'r1_mean',np.mean(np.array(r1diff)),'r1_std',np.std(np.array(r1diff)))
    # print('r0>10',r0,'r0_mean',np.mean(np.array(r0diff)),'r0_std',np.std(np.array(r0diff)))
    # print('all>10',r0+r1,'all_mean',np.mean(np.array(alldiff)),'all_std',np.std(np.array(alldiff)))
           
    data={'root':imglist,'gridscore':gtlist,'netres':reslist}  #,'diff':alldiff 'siminorm':ssimnormlist,'hamnorm':hamnormlist,'simi_labelnew':labelnewlist
    df=pd.DataFrame(data)
    df.to_csv(web_dir+'/out_'+opt.phase+'_all.csv',index=False)
    
    return reslist
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    test_frequency = 1
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    save_img = 0
    
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    print(web_dir)
    save_data_path = web_dir
    na = web_dir[0].split("/")[:]
    # print(image_dir)
    for i_na in range(0, len(na)):
        save_data_path = save_data_path + "/" + na[i_na]
        if not os.path.isdir(save_data_path):
            os.mkdir(save_data_path)
    # import sys
    # sys.exit()
    # frr
    model.eval()
    opt.phase='far'
    farlist=_test(opt,model)
    opt.phase='frr'
    frrlist=_test(opt,model)

    

    # plot
    frrnumpy=np.array(frrlist)
    farnumpy=np.array(farlist)
    # alldata=np.c_[resnumpy,gtnumpy]
    # print(alldata.shape)
    plt.figure()
    plt.title('res')
    plt.hist(frrnumpy,bins=65, rwidth=0.8, range=(55,315),label='Frr', align='left',alpha=0.5)
    plt.hist(farnumpy,bins=65, rwidth=0.8, range=(55,315), label='Far', align='left',alpha=0.5)
    # plt.hist(alldata,bins=65,color=['c','r'], rwidth=0.5, range=(55,315), label=['net_res','gt_label'], align='left',alpha=0.5,stacked=False)
    plt.legend()
    plt.savefig(web_dir+'/frr_far_res.png')
    #judgedice_save_path = web_dir + "/warning"
    #if opt.area:
        #DiceJudge(work_dir,judgedice_save_path)
    #else:
        #AreaJudge(work_dir,judgedice_save_path,)
    # df = pd.DataFrame(contents, columns=['image_path', 'image_name'])
    # df.to_csv('/hdd/file-input/jht/4_bufen_results/shouzhi_results/低温/128_info.csv', index=False, encoding='utf_8_sig')
