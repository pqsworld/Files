增强融合图训练：
config: magicpoint_shapes_pair.yaml
dataset: 'SyntheticDataset_gaussian_cv2_enhance'
front_end_model: 'Train_model_heatmap_cv2'（更小模型）



**C测试**

1. 修改*setting.py* 中的地址  *DATA_PATH = '/hdd/file-input/qint/superpoint_02_unsuperpoint/datasets/C_test'* ，里面存储了一张bmp图片供测试；

2. 修改 *configs/magicpoint_finger_test.yaml* ，注释掉*labels: 'datasets/pre_finger/finger/points'*，禁止读取标签数据；（这一步应该不用进行）

3. "args":[

   ​        "export_detector",

   ​        "configs/magicpoint_finger_test.yaml",

   ​        "test_magicpoint_finally",

   ​        "--outputImg",

   ​        *// "--PR" // 关掉精确度计算*

   ​      ]

完成后即可进行C测试。