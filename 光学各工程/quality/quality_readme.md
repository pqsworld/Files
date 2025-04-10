[toc]

# 质量网络交接文档

## 1.概述

质量网络输入base图，输出质量分数，使用mnv3作为生成器输出质量分数。

## 2.相关路径

| col1       | 服务器       |                                                                                                                |
| ---------- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| 参考原项目 | 172.29.5.203 | /hdd/share/quality/optic_quality/test.py                                                                       |
| 当前路径   | 172.29.5.52  | E:\share\lif\光学各工程\quality                                                                                |
| 训练代码   | 172.29.5.52  | E:\share\lif\光学各工程\quality\\train.py                                                                      |
| 测试代码   | 172.29.5.52  | E:\share\lif\光学各工程\quality\\test.py                                                                       |
| 训练集     | 172.29.5.203 | /hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/train4<br />（同目录下其他数据集可作参考） |
| 测试集     | 172.29.5.203 | /hdd/share/quality/optic_quality/datasets/lif_nvwanew<br />（同目录下其他数据集可作参考）                      |
| pth        | 172.29.5.203 | /hdd/share/quality/checkpoints                                                                                 |

## 3.运行代码

#### 3.1.vscode环境

- 训练
  特别地，c160r96指裁剪到**160x160**后resize到**96x96**

```json
    {
      "name": "Python: 光学质量训练.py",
      "type": "python",
      "request": "launch",
      "env": { "CUDA_VISIBLE_DEVICES": "0,1,5,6,7" },
      "envFile": "${workspaceRoot}/.env",
      "program": "/hdd/share/quality/optic_quality/train.py",

      "console": "integratedTerminal",
      "args": [
        // "-r=true",
        "--dataroot",     "/hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/train4",
        "--name",         "35-0re_c160r96_train7_l5b1k",
        "--gpu_ids",      "0,1,5,6,7",
        "--model",        "pix2pix",
        "--direction",    "AtoB",
        "--batch_size",   "1024",
        // "--crop_size",    "180",
        "--phase",        "train",
        "--load_size_h",  "96",
        "--load_size_w",  "96",
        "--input_nc",     "1",
        "--output_nc",    "1",
        "--netG",         "bufen5",
        "--init_type",    "kaiming",
        "--ndf",          "4",
        "--ngf",          "4",
        "--epoch=1100",
        // "--n_epochs_decay","1000",
        // "--continue_train",
        // "--transform",    "train",
        // "--transform",    "c180r96",
        // "--transform",    "r108c96",
        "--transform",    "c160r96",
        // "--lr_policy","plateau",
        // "--gpu_ids=-1",
        // "--phase=valid",

      ]
    },
```

- 测试
  特别地，测试推理中直接resize到96x96

```json
    {
      "name": "Python: 光学测试.py",
      "type": "python",
      "request": "launch",
      "env": { "CUDA_VISIBLE_DEVICES": "0,1,5,6,7" },
      "envFile": "${workspaceRoot}/.env",
      "program": "/hdd/share/quality/optic_quality/test.py",

      "console": "integratedTerminal",
      "args": [
        // "-r=true",
        "--dataroot",     "/hdd/share/quality/optic_quality/datasets/lif_nvwanew",
        "--name",        "34-base_c160r96_train7_l5b1k",
        "--model",        "pix2pix",
        "--direction",    "AtoB",
        "--batch_size",   "1",
        "--load_size_h",  "96",
        "--load_size_w",  "96",
        // "--crop_size",    "180"z,
        "--input_nc",     "1",
        "--output_nc",    "1",
        "--netG",         "bufen5",
        "--ndf",          "4",
        "--ngf",          "4",
        "--epoch=1000",
        "--gpu_ids=-1",
        "--phase=valid",
      ]
    },
```

#### 3.2 cmd环境

- 训练

  ```shell
  python3 /hdd/share/quality/optic_quality/train.py --dataroot /hdd/share/quality/optic_quality/RankIQA-master/RankIQA-master/data/train4 --name 35-0re_c160r96_train7_l5b1k --gpu_ids 0,1,5,6,7 --model pix2pix --direction AtoB --batch_size 1024 --phase train --load_size_h 96 --load_size_w 96 --input_nc 1 --output_nc 1 --netG bufen5 --init_type kaiming --ndf 4 --ngf 4 --epoch=1100 --transform c160r96
  ```
- 测试

  ```shell
  python3 /hdd/share/quality/optic_quality/test.py --dataroot /hdd/share/quality/optic_quality/datasets/lif_nvwanew --name 34-base_c160r96_train7_l5b1k --model pix2pix --direction AtoB --batch_size 1 --load_size_h 96 --load_size_w 96 --input_nc 1 --output_nc 1 --netG bufen5 --ndf 4 --ngf 4 --epoch=1000 --gpu_ids=-1 --phase=valid
  ```

## 4.其他

当前网络分数参考，仅供参考。另有文件夹参考文档有具图：E:\share\lif\光学各工程\quality\\quality参考图像.xlsx

| col1 | col2                               |
| ---- | ---------------------------------- |
| 0    | 全黑全白                           |
| -10  | 无纹路                             |
| -20  | 纹路不清晰 曝光异常 采图失真       |
| -30  | 小部分纹路清晰 曝光异常 采图失真   |
| -40  | 小部分纹路清晰 曝光异常 采图不失真 |
| -50  | 中部分纹路清晰 曝光异常 采图不失真 |
| -60  | 大部分纹路清晰 曝光异常 采图不失真 |
| -70  | 纹路清晰 曝光异常 采图不失真       |
| -80  | 纹路清晰 曝光较正常 采图不失真     |
| -90  | 纹路清晰 曝光正常 采图不失真       |
| -99  | 纹路很清晰 曝光正常 采图不失真     |
| 100  | 训练集正例                         |

阈值段：

| 30分以下 不可用 |
| --------------- |
| 40-70 质量差    |
| 80 曝光问题     |
| 90 质量优       |
