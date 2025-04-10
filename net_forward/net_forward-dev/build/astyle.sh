#!/bin/bash

MY_PATH=`pwd`
BRANCH_PATH=..
CODE_PATH=$MY_PATH/$BRANCH_PATH

EXT_ORIG="*.orig"

#folder_array=(
#"identify_api"
#)




cd anycode_to_utf8
rm  to_utf8
make
#for folder in  ${folder_array[@]}
for folder in  $CODE_PATH
do
  ./path_to_utf8.sh $folder
done
cd -

#开源代码地址
#url = https://github.com/chenhongjun/anycode_to_utf8.git
if [ "$1" = "rm" ]
then
  for folder in  $CODE_PATH
  do
    echo "rm -vf $folder/$EXT_ORIG"
    rm -vf $folder/$EXT_ORIG
  done
fi

