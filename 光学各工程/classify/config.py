'''Fingerprint ASP Config.'''

#traindir = '/ssd/yangwc/MobileNetV3-master/data/splitData/train'
#valdir = '/ssd/yangwc/MobileNetV3-master/data/splitData/test'
#tstdir = '/ssd/yangwc/MobileNetV3-master/data/splitData/test'

#traindir = '/hdd/file-input/wangb/classify/code/comparenet/data/train_2023_11_23_07_45'
#valdir = '/hdd/file-input/wangb/classify/code/comparenet/data/test_2023_11_23_07_45'
#tstdir = '/hdd/file-input/wangb/classify/code/comparenet/data/test_2023_11_23_07_45'


#traindir = './data/train/1'
#valdir = './data/test/1'
#tstdir = './data/test/1'

#traindir = './data/train_2023_12_29_03_16'



#traindir = './data/train_net'#train_net'
#valdir = './data/test_2023_12_29_03_16'#test_2023_12_29_03_16'
#tstdir = './data/test_2023_12_29_03_16'#test_2023_12_29_03_16'



#path_txt = "./data/6195/6195database.train/Train"
#traindir = './data/6195/train_crop'
#traindir = './data/train_add'
#valdir = './data/6195/test_crop'
#tstdir = './data/6195/test_crop'

#traindir = './data/train_bc'
#valdir = './data/test_2023_12_29_03_16'
#tstdir = './data/test_2023_12_29_03_16'


traindir = '../datasets/6195/train_bc_crop_66-simiall'#test_2023_12_29_03_16_crop_66'#train_bc_crop_66'
valdir = '/hdd/file-input/wangb/classify/compare/data/test_crop_66'
tstdir = '/hdd/file-input/wangb/classify/compare/data/test_crop_66'


#valid
ratio = 0.7
channel = 1

# train data setting
input_size = (118,32)
num_worker = 12
lr = 0.01
batch_size = 1024
epoch = 400

