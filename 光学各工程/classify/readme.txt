训练模型：
train.py（电容的）或者train_st0.py（光学的）


数据：
209服务器上 share目录下面

电容的：
训练集：/ssd/share/liugq/datas/620/train620
测试集：/ssd/share/liugq/datas/test620

光学数据：
训练集：/ssd/share/liugq/ttl/honor_124/ttl16_train
测试集：/ssd/share/liugq/ttl/honor_124/ttl16_test

##初始训练
python3  train_st0.py  --name class6195_eh620_w3   --gpu_ids 2  --lr 0.01 --epoch 400 --batchsize 3000  --checkpoints_dir ./checkpoints --model mnv_small1  --optim_choose Adam --data_train   /ssd/share/liugq/datas/620/train620  --data_test /ssd/share/liugq/datas/test620 --imsize 66  --save_epoch True     --inputchannels 2 --crop_flag  True --width 66  --enlarge_flag True
## 迭代训练
python3  train_st0.py  --name class6195_eh620_w3   --gpu_ids 2  --lr 0.01 --epoch 400 --batchsize 3000  --checkpoints_dir ./checkpoints --model mnv_small1  --optim_choose Adam --data_train   /ssd/share/liugq/datas/620/train620  --data_test /ssd/share/liugq/datas/test620 --imsize 66  --save_epoch True     --inputchannels 2 --crop_flag  True --width 66  --enlarge_flag True --loadmodel_flag True
### 迭代训练+新训练集
python3  train_st0.py  --name class6195_eh620_w3   --gpu_ids 2  --lr 0.01 --epoch 400 --batchsize 3000  --checkpoints_dir ./checkpoints --model mnv_small1  --optim_choose Adam --data_train   /ssd/share/liugq/datas/620/train620  --data_test /ssd/share/liugq/datas/test620 --imsize 66  --save_epoch True     --inputchannels 2 --crop_flag  True --width 66  --enlarge_flag True --loadmodel_flag True --add_smallarea True  --smallroot /ssd/share/liugq/datas/train_bc-simi 

光学的：
python3  train_st0.py  --name smallttl_i5  --gpu_ids 4  --lr 0.01 --epoch 400 --batchsize 2048  --checkpoints_dir ./checkpoints --model mnv_small1 --data_train  /ssd/share/liugq/ttl/honor_124/ttl16_train  --data_test  /ssd/share/liugq/ttl/honor_124/ttl16_test --optim_choose Adam  --imsize 124 --save_epoch True     --inputchannels 2 --crop_flag  True --width 124  --enlarge_flag True  --jitter True  --loadmodel_flag True --add_smallarea True  --smallroot /ssd/share/liugq/ttl/honor_124/e16_train



测试代码：
test.py 这个看准确率
confirm_score_mobilenet_fast_table.py 这个刷阈值表

生成对位图数据：
generate_tool/ttl_enhpth_computer_score.py


已经训练好的模型：
./pths/


测试模型性能：
confirm_score_mobilenet_fast_table.py

提取模型参数：
get_parametershort.py




有效的优化策略：
1. 数据扩增 （这个既可以是常规的数据扩增方法，也可以根据实际需要编写特别的数据扩增方法，比如trans抖动、中心抖动、单通道抖动等，这个在train_sto.py都有）

2.优化网络结构（加入一些特殊的结构可以很好的提升性能）

3.迭代优化（多迭代几次，每次训练的时候初始模型用前一次的最佳模型）

4.教师监督（使用较大的模型训练，然后用大模型监督小模型，这个需要适当调整教师loss的比例，不能加入太大，否则会起到负面作用）

5.加入更多的实际数据
   对于特别差的图发现分类器性能很难提升，可以采集一些相关的差图，然后人工筛选一下加入到训练集中，这个可以很好的提升分类器相关的性能。比如加入了beta差图数据