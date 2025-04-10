#!/bin/bash
echo "build scipts for jenkins builds"
echo $1
echo $2

sl_jenkins_path=`pwd`

rm -rf *.xz
rm -rf gitlab_artifact

#if ["$BRANCH_NAME"="release_hw_7015"];then
./build_all  $1 $2
if [ $? != 0 ]
then
echo "build_all fail"
exit -1
else
echo "build_all success"

mkdir gitlab_artifact
ls -aht *.xz|head -n 1|xargs -i cp {} gitlab_artifact
fi
exit 0
