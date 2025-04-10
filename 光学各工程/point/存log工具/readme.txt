一、图库命名标准
场景->人->手指


二、
1、工程sl_algorithm

2、存图和log的宏
GET_ALL_TRANS 

3、main函数给路径，编译
TestFrr_After_Resi("F:\\zhangderong\\data\\smallTTL\\124x124_bin\\bng\\DW", SL_SAVE_LEVEL_0);
路径最好用自己的，存图的时候会改变图库内容

编译之后
F:\zhangderong\code_deep\00_prepare_train_test_data\smallTTL_DescSave\sl_algorithm\x64\Release 路径下有执行文件可以重命名多个跑

跑完之后会生成*_MatchTrans_Frr000_*.txt

4、调用get_trans.py处理txt