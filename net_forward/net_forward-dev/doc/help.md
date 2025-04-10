[TOC]
### 1. android 环境切换

我们的android，都是同步的软件部门的，有问题可以直接咨询软件。

```bash
cd ~/back_tool/aosp
cisco@silead-1:~/back_tool/aosp$ git branch 
* android_6
  android_7
  master
cisco@silead-1:~/back_tool/aosp$ git checkout android_6
```

### 2. 更改android环境代码 

android_6 用fix-aosp-6-compile-env.diff 这个patch,

```bash
cisco@silead-1:~/back_tool/aosp$ cat fix-aosp-6-compile-env.diff | patch -p1
```

39服务器需要另外改一下这。

```bash
yangwc@suanfa:~/tool/aosp_6$ git diff art/build/Android.common_build.mk
diff --git a/art/build/Android.common_build.mk b/art/build/Android.common_build.mk
index b84154b..255c560 100644
--- a/art/build/Android.common_build.mk
+++ b/art/build/Android.common_build.mk
@@ -72,7 +72,7 @@ ART_TARGET_CFLAGS :=

 # Host.
 ART_HOST_CLANG := false
-ifneq ($(WITHOUT_HOST_CLANG),true)
+ifeq ($(WITHOUT_HOST_CLANG),true)
   # By default, host builds use clang for better warnings.
   ART_HOST_CLANG := true
 endif
yangwc@suanfa:~/tool/aosp_6$
 ```
android_7 用fix-aosp-7-compile-env.diff这个patch。


### 3. 配置Java环境

android_6 用

```bash
cisco@silead-1:~/back_tool/aosp$ export PATH=/usr/lib/jvm/java-1.7.0-openjdk-amd64/bin:$PATH
```
android_7 用

```bash
cisco@silead-1:~/back_tool/aosp$ export PATH=/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin:$PATH
```
android_6 android_7 都需要这一步配置一下java 环境

```bash
cisco@silead-1:~/back_tool/aosp$ . /apps/java8.sh
```


### 4. 编译android

删除之前android编译生成的文件, 配置环境并且编译。编译前可以清空一下vendor下的所有内容，

因为 make 的时候可能会编译到vendor下的代码，导致编译失败。

```bash
cisco@silead-1:~/back_tool/aosp$ rm -rf vendor/*
cisco@silead-1:~/back_tool/aosp$ rm -r out/*
cisco@silead-1:~/back_tool/aosp$ source ./build/envsetup.sh
cisco@silead-1:~/back_tool/aosp$ lunch aosp_arm64-user
cisco@silead-1:~/back_tool/aosp$ make -j32
```

