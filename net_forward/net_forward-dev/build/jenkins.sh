#!/bin/bash
echo "build scipts for jenkins builds"
echo $1
echo $2

sl_jenkins_path=`pwd`

#if ["$BRANCH_NAME"="release_hw_7015"];then
./build_all  $1 $2
if [ $? != 0 ]
then
echo "build_all fail"
exit 1
else
echo "build_all success"
fi

echo	"cppcheck start"
cd ..
rm doc/cppcheck_net_forward.xml
cppcheck -j 16 --language=c --addon=cert --inconclusive --xml --xml-version=2  --enable=all  . 2>doc/cppcheck_net_forward.xml
cp -f doc/cppcheck_net_forward.xml /data/guest/LIB_NET网络库/cppcheck_net_forward.xml

current_short_head=`git rev-parse --short HEAD`

echo "cppcheck end"
exit 0