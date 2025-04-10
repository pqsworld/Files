  #!/bin/sh
BRANCH_NAME=$3
MY_PATH=$4

LIB_NAME_1=libsilfp_algo_net.lib

if [ ! -d "$BRANCH_NAME/qsee" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/qsee
 mkdir $MY_PATH/$BRANCH_NAME/qsee/qsee1
 mkdir $MY_PATH/$BRANCH_NAME/qsee/qsee2
 mkdir $MY_PATH/$BRANCH_NAME/qsee/qsee3
 echo "$BRANCH_NAME/qsee create"
else
 rm $MY_PATH/$BRANCH_NAME/qsee/qsee1/*
 rm $MY_PATH/$BRANCH_NAME/qsee/qsee2/*
 rm $MY_PATH/$BRANCH_NAME/qsee/qsee3/*
 echo "$BRANCH_NAME/android_memtest exist"
fi

if [ ! -d "$BRANCH_NAME/qsee4" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/qsee4
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib64

 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/sec_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x2
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib32/x2

 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib64/x2
 mkdir $MY_PATH/$BRANCH_NAME/qsee4/lib64/sec_x1
 echo "$BRANCH_NAME/qsee4 create"
else
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/sec_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x2/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib32/x2/*

 rm $MY_PATH/$BRANCH_NAME/qsee4/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib64/x2/*
 rm $MY_PATH/$BRANCH_NAME/qsee4/lib64/sec_x1/*
 echo "$BRANCH_NAME/qsee4 exist"
fi

if [ ! -d "$BRANCH_NAME/tee-lib" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib64

 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/sec_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib32/sec_x1

 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-lib/lib64/sec_x1
 echo "$BRANCH_NAME/tee-lib create"
else
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x1*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x2/*

 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x2/*
 echo "$BRANCH_NAME/tee-lib exist"
fi

if [ ! -d "$BRANCH_NAME/tee-VI" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib64

 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x2

 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x2
 echo "$BRANCH_NAME/tee-VI create"
else
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x2/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x2/*

 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x2/*
 echo "$BRANCH_NAME/tee-VI exist"
fi

if [ ! -d "$BRANCH_NAME/tee-VII" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/tee-VII
 mkdir $MY_PATH/$BRANCH_NAME/tee-VII/lib64
 mkdir $MY_PATH/$BRANCH_NAME/tee-VII/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x2
 echo "$BRANCH_NAME/tee-VII create"
else
 rm $MY_PATH/$BRANCH_NAME/tee-VII/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x2/*
 echo "$BRANCH_NAME/tee-VII exist"
fi

if [ ! -d "$BRANCH_NAME/tee-VIII" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib64

 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x2

 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x2
 echo "$BRANCH_NAME/tee-VIII create"
else
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x2/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x2/*

 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x2/*
 echo "$BRANCH_NAME/tee-VIII exist"
fi

if [ ! -d "$BRANCH_NAME/tee-IX" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib64

 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x2
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x2

 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x2
 echo "$BRANCH_NAME/tee-IX create"
else
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x2/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x2/*

 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x2/*
 echo "$BRANCH_NAME/tee-IX exist"
fi

if [ ! -d "$BRANCH_NAME/windows" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/windows
 mkdir $MY_PATH/$BRANCH_NAME/windows/lib32
 mkdir $MY_PATH/$BRANCH_NAME/windows/lib64
 echo "$BRANCH_NAME/windows create"
else
 rm $MY_PATH/$BRANCH_NAME/windows/lib32/*
 rm $MY_PATH/$BRANCH_NAME/windows/lib64/*
 echo "$BRANCH_NAME/windows exist"
fi

if [ ! -d "$BRANCH_NAME/clang13" ]
then
 mkdir $MY_PATH/$BRANCH_NAME/clang13
 mkdir $MY_PATH/$BRANCH_NAME/clang13/lib64

 mkdir $MY_PATH/$BRANCH_NAME/clang13/lib64/ei_x1
 mkdir $MY_PATH/$BRANCH_NAME/clang13/lib64/x1
 mkdir $MY_PATH/$BRANCH_NAME/clang13/lib64/x2
 mkdir $MY_PATH/$BRANCH_NAME/clang13/lib64/x3
 echo "$BRANCH_NAME/clang13 create"
else
 rm $MY_PATH/$BRANCH_NAME/clang13/lib64/ei_x1/*
 rm $MY_PATH/$BRANCH_NAME/clang13/lib64/x1/*
 rm $MY_PATH/$BRANCH_NAME/clang13/lib64/x2/*
 rm $MY_PATH/$BRANCH_NAME/clang13/lib64/x3/*
 echo "$BRANCH_NAME/clang13 exist"
fi
# if [ ! -d "$BRANCH_NAME/linux-lib" ]
# then
#  mkdir $MY_PATH/$BRANCH_NAME/linux-lib
#  mkdir $MY_PATH/$BRANCH_NAME/linux-lib/lib32
#  mkdir $MY_PATH/$BRANCH_NAME/linux-lib/lib64
#  echo "$BRANCH_NAME/linux-lib create"
# else
#  rm $MY_PATH/$BRANCH_NAME/linux-lib/lib32/*
#  rm $MY_PATH/$BRANCH_NAME/linux-lib/lib64/*
#  echo "$BRANCH_NAME/linux-lib exist"
# fi


echo "$MY_PATH/output/qsee/qsee1/$LIB_NAME_1"
#if [ "$1" = "api" -o "$1" = "all" ]; then
if [ -f "$MY_PATH/output/qsee4/lib32/ei_neon_x1/$LIB_NAME_1" ]; then
# if [ -f "$MY_PATH/output/qsee/qsee1/$LIB_NAME_1" ]; then
#  cp -r $MY_PATH/output/qsee/qsee1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee/qsee1/
#  cp -r $MY_PATH/output/qsee/qsee2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee/qsee2/
#  cp -r $MY_PATH/output/qsee/qsee3/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee/qsee3/

 cp -r $MY_PATH/output/qsee4/lib32/ei_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_neon_x1/
 cp -r $MY_PATH/output/qsee4/lib32/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/ei_x1/
 cp -r $MY_PATH/output/qsee4/lib32/neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x1/
 cp -r $MY_PATH/output/qsee4/lib32/sec_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/sec_neon_x1/
 cp -r $MY_PATH/output/qsee4/lib32/neon_x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/neon_x2/
 cp -r $MY_PATH/output/qsee4/lib32/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/x1/
 cp -r $MY_PATH/output/qsee4/lib32/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib32/x2/
 cp -r $MY_PATH/output/qsee4/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib64/ei_x1/
 cp -r $MY_PATH/output/qsee4/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib64/x1/
 cp -r $MY_PATH/output/qsee4/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib64/x2/
 cp -r $MY_PATH/output/qsee4/lib64/sec_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/qsee4/lib64/sec_x1/

 cp -r $MY_PATH/output/tee-lib/lib32/ei_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_neon_x1/
 cp -r $MY_PATH/output/tee-lib/lib32/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/ei_x1/
 cp -r $MY_PATH/output/tee-lib/lib32/neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x1/
 cp -r $MY_PATH/output/tee-lib/lib32/neon_x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/neon_x2/
 cp -r $MY_PATH/output/tee-lib/lib32/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x1/
 cp -r $MY_PATH/output/tee-lib/lib32/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/x2/
 cp -r $MY_PATH/output/tee-lib/lib32/sec_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/sec_x1/
 cp -r $MY_PATH/output/tee-lib/lib32/sec_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib32/sec_neon_x1/
 cp -r $MY_PATH/output/tee-lib/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib64/ei_x1/
 cp -r $MY_PATH/output/tee-lib/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x1/
 cp -r $MY_PATH/output/tee-lib/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib64/x2/
 cp -r $MY_PATH/output/tee-lib/lib64/sec_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-lib/lib64/sec_x1/

 cp -r $MY_PATH/output/tee-VI/lib32/ei_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_neon_x1/
 cp -r $MY_PATH/output/tee-VI/lib32/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/ei_x1/
 cp -r $MY_PATH/output/tee-VI/lib32/neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x1/
 cp -r $MY_PATH/output/tee-VI/lib32/neon_x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/neon_x2/
 cp -r $MY_PATH/output/tee-VI/lib32/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x1/
 cp -r $MY_PATH/output/tee-VI/lib32/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib32/x2/
 cp -r $MY_PATH/output/tee-VI/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib64/ei_x1/
 cp -r $MY_PATH/output/tee-VI/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x1/
 cp -r $MY_PATH/output/tee-VI/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VI/lib64/x2/

 cp -r $MY_PATH/output/tee-VII/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VII/lib64/ei_x1/
 cp -r $MY_PATH/output/tee-VII/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x1/
 cp -r $MY_PATH/output/tee-VII/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VII/lib64/x2/

 cp -r $MY_PATH/output/tee-VIII/lib32/ei_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_neon_x1/
 cp -r $MY_PATH/output/tee-VIII/lib32/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/ei_x1/
 cp -r $MY_PATH/output/tee-VIII/lib32/neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x1/
 cp -r $MY_PATH/output/tee-VIII/lib32/neon_x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/neon_x2/
 cp -r $MY_PATH/output/tee-VIII/lib32/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x1/
 cp -r $MY_PATH/output/tee-VIII/lib32/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib32/x2/
 cp -r $MY_PATH/output/tee-VIII/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/ei_x1/
 cp -r $MY_PATH/output/tee-VIII/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x1/
 cp -r $MY_PATH/output/tee-VIII/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-VIII/lib64/x2/

 cp -r $MY_PATH/output/tee-IX/lib32/ei_neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_neon_x1/
 cp -r $MY_PATH/output/tee-IX/lib32/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/ei_x1/
 cp -r $MY_PATH/output/tee-IX/lib32/neon_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x1/
 cp -r $MY_PATH/output/tee-IX/lib32/neon_x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/neon_x2/
 cp -r $MY_PATH/output/tee-IX/lib32/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x1/
 cp -r $MY_PATH/output/tee-IX/lib32/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib32/x2/
 cp -r $MY_PATH/output/tee-IX/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib64/ei_x1/
 cp -r $MY_PATH/output/tee-IX/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x1/
 cp -r $MY_PATH/output/tee-IX/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/tee-IX/lib64/x2/

 cp -r $MY_PATH/output/windows/lib32/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/windows/lib32/
 cp -r $MY_PATH/output/windows/lib64/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/windows/lib64/

 cp -r $MY_PATH/output/clang13/lib64/ei_x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/clang13/lib64/ei_x1/
 cp -r $MY_PATH/output/clang13/lib64/x1/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/clang13/lib64/x1/
 cp -r $MY_PATH/output/clang13/lib64/x2/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/clang13/lib64/x2/
 cp -r $MY_PATH/output/clang13/lib64/x3/$LIB_NAME_1 $MY_PATH/$BRANCH_NAME/clang13/lib64/x3/
fi

#rm -rf $MY_PATH/$BRANCH_NAME/.git
#cp $MY_PATH/alg/libfftw3f_32.a $MY_PATH/$BRANCH_NAME/quickfft/arm/
#cp /home/tool/qsee3-build-samp/output/qsee2 /home/tool/$BRANCH_NAME
#cp /home/tool/qsee3-build-samp/output/tee-lib /home/tool/$BRANCH_NAME
#cp /home/tool/qsee3-build-samp/output/qsee4 /home/tool/$BRANCH_NAME

PREFIX=$5
#EXT=tar.bz2
EXT=tar.xz
TAR_NAME=$PREFIX-$(date "+%Y%m%d%H%M%S")

tar Jcvf $TAR_NAME.$EXT $BRANCH_NAME/*

tree $BRANCH_NAME
echo "$TAR_NAME.$EXT build success."
if [ -z "$6" ]; then
 echo "is in the $4/$TAR_NAME.$EXT directory"
else
 mv $TAR_NAME.$EXT $6
 echo "$TAR_NAME.$EXT is in the $6 directory"
fi

#cp -f LIB_NET/windows/lib32/libsilfp_algo_net.lib /data/guest/LIB_NET网络库/
{
  ls -aht *.xz|head -n 1|xargs -i cp {} /data/ftp/output/LIB_NET
  echo "copied to \\172.29.4.220\ftp\output\LIB_NET"
} || {
  echo "current user have no access to 220:/dafa/ftp/output/LIBNET. cp FAILED."
}
#ls -aht *.xz|head -n 1|xargs -i cp {} /data/guest/LIB_NET网络库/
