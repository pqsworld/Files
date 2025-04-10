_您现在位于：Test分支_

[toc]


_Generated from "net forward" by Stable Diffusion V2._

---

# Net Forward

> 网络库编译项目

本项目实现归一化的网络单独编库。主要负责编译**增强、描述子、mask、扩边、防伪、误触、异物**等深度算法库。


## 2. Struct

### 2.1 Branch


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
