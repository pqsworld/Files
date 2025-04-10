_您现在位于：Test分支_

[toc]
<!-- # Net Forward [![220](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme) -->

<img src="/doc/sd.png" />

_Generated from "net forward" by Stable Diffusion V2._

---

# Net Forward

> 网络库编译项目。DL.Algo.Silead.Gigadevice.

本项目实现归一化的网络单独编库。主要负责编译**增强、描述子、mask、扩边、防伪、误触、异物**等深度算法库。
合入了jenkins的自动静态扫描（cppcheck）和自动编库(CD)模块。

master,dev,release分支禁止提交。
开发人员新建分支后push远程。
在远程申请合并(**PullRequest/MergeRequest**)

申请合并后触发编库，reviewers审核后合入主干。
版本和扫描结果保存在 **文件浏览器：\\\172.29.4.220\ftp\output**。

## 1. Release History

### 1.1 项目release

| 版本  |   日期   | 描述                                    | 补充描述             |
| :---: | :------: | :-------------------------------------- | -------------------- |
| 0.0.6 | 23-02-02 | Add: spd网络 |Debug：解耦误触和防伪|
| 0.0.5 | 23-01-11 | Add: 增加版本号结构 |Debug：修复NEON功能(enhance.c)  |
| 0.0.4 | 22-12-20 | Debug: 修复warnings问题 |push时发送邮件给reviewer  |
| 0.0.3 | 22-12-15 | Debug: 修改astyle, 修复编译数学库的问题 |                      |
| 0.0.2 | 22-12-13 | ADD：增加readme文档                     | FIX: linux编译链修复 |
| 0.0.1 | 22-12-10 | INIT：初始化                            |                      |

### 1.2 算法release：主版本已更新230202002 异物/增强

| 算法   |描述| 版本                       | 备注 |
| ------ | ----     |--------------- | ---- |
|主版本  |230202002|                  |      |
| 描述子 | Descriptor | 23011100000   |      |
| 增强   | Enhance    |230119302      |加round|
| 扩边   | Exp        |23011100000    |      |
| mask   |Mask        |23011100000    |      |
| 误触   | Mistouch   |23011100000    |      |
| 防伪   |Spoof       |23011100000    |      |
| 异物   |Spd         |221219301      |      |
| 质量   |Quality    |303239301      |      |

## 2. Struct

### 2.1 Branch

* **master**
  <u>主分支</u>，**developer提交申请后由reviewer合入**。维持公共主干，common代码有修改才会合入。
* **dev**
  <u>开发分支</u>，developer合入。可用于开发和编库。
* **release**
  <u>版本分支</u>，**developer提交申请后由reviewer合入**。用于出版本。
* **test**
  <u>测试分支</u>，developer合入。开发者推送后自动进jenkins测试。

### 2.2 Code

* [build](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/build) - 编译目录。一般不需要修改，具体编译命令见下文。
  * [build/LIB_NET](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/build/) - 编译出的网络库所在目录。最后会打包为.tar.xz文件。LIB_NET不会被gitlab提交，仅在本地生存。
  * [build/*.tar.cz](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/build/) - 编译出的网络库打包文件。xz文件都不会被gitlab提交，仅在本地生存。

* [doc](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/doc) - 文档目录。一般不需要修改。TODO:文档移植

* [net](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net) - 网络主要代码。
  * [net/net_param](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net/net_param) - 参数目录。存放`扩边(net_exp)`、`描述子(desc_patch)` 、`增强(enhance)`、`mask`、`误触(mistouch)`、`防伪(spoof)`网络的权重参数。
  * [net/net_param/para_struct_enhance.h](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net/net_param) - 网络结构的定义。
  * [net/net_param/struct_enhance.h](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net/net_param) - 网络结构的声明。
  * [net/test](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net/test) - 测试单元目录。具体测试命令见下文。
  * [net/net_enhance.h](http://172.29.4.220:8000/silead-algo-dl/net_forward/-/tree/dev/net/test/net_enhance.h) - 深度函数接口。目前包括扩边、mask、增强、描述子。TODO：解耦为四个函数。

## 3. Builds

### 3.1 编库命令

```shell
//执行格式脚本，以保证和原格式一致，可略过
$ cd net_forward
$ ./astyle.sh ./

//6193可以换成其他芯片/square
$ cd build
$ ./build_all 版本号 6193

```

编译好的lib库存放在**build/out**，会自动打包为.xz文件。
并自动复制到**guest@172.29.4.220: /LIBNET网络库文件夹**下。

----

### 3.2 测试命令

```shell
//清除原有lib下.o文件
$ rm -rf ./lib/* && ls ./lib/

//复制linux静态库到lib文件夹下
$ cp -f ../../build/output/linux-x86/lib64/libsilfp_algo_net.lib ./lib/ && l ./lib/

//解开lib内.o文件
$ cd lib/ && ar -x libsilfp_algo_net.lib && l && cd ..

//gcc编译main.c为main_001可执行文件，其中添加相关.o文件为中间件
$ gcc main.c  lib/*.o  -o main_001
$ ./main_001
```

## 4. Contribute/Reviewer

| 描述子 | 增强   | mask | 扩边   | 防伪 | 误触 | 维护 |
| ------ | ------ | ---- | ------ | ---- | ---- | ---- |
| 叶杨   | 刘功琴 | 王奔 | 刘功琴 | 林杜 | 林杜 | 潘琪 |

## 5. License

Copyright (c) 2005-2022 **GigaDevice/Silead** Inc.

All rights reserved

The present software is the confidential and proprietary information of [GigaDevice](https://www.gigadevice.com/)/[Silead Inc](http://www.sileadinc.com/).

You shall not disclose the present software and shall use it only in accordance with the terms of the license agreement you entered into with [GigaDevice](https://www.gigadevice.com/)/[Silead Inc](http://www.sileadinc.com/).

This software may be subject to export or import laws in certain countries.
