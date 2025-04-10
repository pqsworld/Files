#!/bin/bash

PATH_S=`pwd`
echo $PATH_S
:<<EOF
rm $PATH_S/output/qsee/qsee1/*
rm $PATH_S/output/qsee/qsee2/*
rm $PATH_S/output/qsee/qsee3/*

rm $PATH_S/output/qsee4/lib32/ei_neon_x1/*
rm $PATH_S/output/qsee4/lib32/ei_x1/*
rm $PATH_S/output/qsee4/lib32/neon_x1/*
rm $PATH_S/output/qsee4/lib32/sec_neon_x1/*
rm $PATH_S/output/qsee4/lib32/neon_x2/*
rm $PATH_S/output/qsee4/lib32/x1/*
rm $PATH_S/output/qsee4/lib32/x2/*

rm $PATH_S/output/qsee4/lib64/ei_x1/*
rm $PATH_S/output/qsee4/lib64/x1/*
rm $PATH_S/output/qsee4/lib64/x2/*
rm $PATH_S/output/qsee4/lib64/sec_x1/*

rm $PATH_S/output/tee-lib/lib32/ei_neon_x1/*
rm $PATH_S/output/tee-lib/lib32/ei_x1/*
rm $PATH_S/output/tee-lib/lib32/neon_x1/*
rm $PATH_S/output/tee-lib/lib32/neon_x2/*
rm $PATH_S/output/tee-lib/lib32/x1/*
rm $PATH_S/output/tee-lib/lib32/x2/*

rm $PATH_S/output/tee-lib/lib64/ei_x1/*
rm $PATH_S/output/tee-lib/lib64/x1/*
rm $PATH_S/output/tee-lib/lib64/x2/*

rm $PATH_S/output/tee-VII/lib64/ei_x1/*
rm $PATH_S/output/tee-VII/lib64/x1/*
rm $PATH_S/output/tee-VII/lib64/x2/*

rm $PATH_S/output/tee-VIII/lib32/ei_neon_x1/*
rm $PATH_S/output/tee-VIII/lib32/ei_x1/*
rm $PATH_S/output/tee-VIII/lib32/neon_x1/*
rm $PATH_S/output/tee-VIII/lib32/neon_x2/*
rm $PATH_S/output/tee-VIII/lib32/x1/*
rm $PATH_S/output/tee-VIII/lib32/x2/*

rm $PATH_S/output/tee-VIII/lib64/ei_x1/*
rm $PATH_S/output/tee-VIII/lib64/x1/*
rm $PATH_S/output/tee-VIII/lib64/x2/*

rm $PATH_S/output/tee-VI/lib32/ei_neon_x1/*
rm $PATH_S/output/tee-VI/lib32/ei_x1/*
rm $PATH_S/output/tee-VI/lib32/neon_x1/*
rm $PATH_S/output/tee-VI/lib32/neon_x2/*
rm $PATH_S/output/tee-VI/lib32/x1/*
rm $PATH_S/output/tee-VI/lib32/x2/*

rm $PATH_S/output/tee-VI/lib64/ei_x1/*
rm $PATH_S/output/tee-VI/lib64/x1/*
rm $PATH_S/output/tee-VI/lib64/x2/*

rm $PATH_S/output/tee-IX/lib32/ei_neon_x1/*
rm $PATH_S/output/tee-IX/lib32/ei_x1/*
rm $PATH_S/output/tee-IX/lib32/neon_x1/*
rm $PATH_S/output/tee-IX/lib32/neon_x2/*
rm $PATH_S/output/tee-IX/lib32/x1/*
rm $PATH_S/output/tee-IX/lib32/x2/*

rm $PATH_S/output/tee-IX/lib64/ei_x1/*
rm $PATH_S/output/tee-IX/lib64/x1/*
rm $PATH_S/output/tee-IX/lib64/x2/*

rm $PATH_S/output/windows/lib32/*
rm $PATH_S/output/windows/lib64/*

rm $PATH_S/output/linux-lib/lib32/*
rm $PATH_S/output/linux-lib/lib64/*

rm $PATH_S/output/clang13/lib64/ei_x1/*
rm $PATH_S/output/clang13/lib64/x1/*
rm $PATH_S/output/clang13/lib64/x2/*
rm $PATH_S/output/clang13/lib64/x3/*

rm -f $PATH_S/LIB_NET-silead*.tar.xz
EOF

rm ~/tool/aosp/out/target/product/generic_arm64/system/lib/*
rm ~/tool/aosp/out/target/product/generic_arm64/system/lib64/*

rm -rf fp-lib-algo-* build-err.log $PATH_S/output
