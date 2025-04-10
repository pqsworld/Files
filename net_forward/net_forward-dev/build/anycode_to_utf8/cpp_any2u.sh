#!/bin/bash

function convert_file() {

# 检查文件是否存在
filename=$1
if [ ! -f $filename ]; then
    echo "$filename isn't exists"
    return
fi

# 转换文件编码为UTF-8
file -i $filename | grep -q "iso-8859-1"
if [ $? -eq 0 ]; then
    echo "iconv -f GBK -t UTF-8 -o $filename".tmp" $filename"
    iconv -f GBK -t UTF-8 -o $filename".tmp" $filename
    mv $filename".tmp" $filename
fi

#使用file -i
./to_utf8 $filename

:<<EOF
 -s4 : 表示缩进4个空格，默认配置；
 -S: 表示switch中case语句的缩进；
 -N: 表示命名空间namespace内的缩进；
 -U: 表示括号内的两头的参数和括号之间不留空格；
 -H: 表示”if”、”for”、”while”等关键字右边增加一个空格；
 -k1: *和&在表示指针和引用类型时，和类型名称并紧，和变量名之间留空格；
-p: 在运算符号(操作符)左右加上空格；
 -P: 在括号两边插入空格；-d只在括号外面插入空格，-D只在里面插入；
 -j: 给每个”if”、”for”、“while”增加大括号；
-D: 在小括号边上增加一个空格；
 -c: 将TAB替换成空格；
 -M: 对定义的参数和变量进行对齐；
 -w: 对宏进行对齐处理；
-Y: C++的注释方式//加上一个空格
--style=ansi: ANSI标准的文件格式，对”{”、”}”另启一行；
--indent=spaces=4: 缩进采用4个空格；
 --add-brackets: 对”if”、”for”、“while”单行的语句增加括号；
--convert-tabs: 强制转换TAB为空格；
--indent-preprocessor: 将preprocessor(#define)等这类预定义的语句，如果有多行时前面填充对齐(是对单语句多行进行填充)；
--align-pointer=type: *、&这类字符靠近类型；
 --align-pointer=name: *、&这类字符靠近变量名字；
--pad-oper: 在操作符号两边增加空格字符；
--pad--header: 在关键字”if”、”for”、”while”等后面增加空格；
--indent-switches: switch case的代码也按照标准缩进方式缩进；
--indent-col1-comments: 如果函数开始后面(“{”后面)第一行是注释，也进行缩进；
--indent=tab: 显示说明使用Tab；
EOF
echo "astyle --style=linux --mode=c -ns4 -q -f -H -p -j -c -M -w -k1 -Y --convert-tabs $filename"
astyle --style=linux --mode=c -ns4 -q -f -H -p -j -c -M -w -k1 -Y --convert-tabs $filename
#使用enca
#echo $filename
#enca -L zh_CN $filename
#enca -L zh_CN -x UTF-8 $filename

# 验证是否为UTF-8
file -i $filename | grep -q "utf-8\|us-ascii"
if [ $? -eq 1 ]; then
    echo "$filename is not utf-8 file"
    return
fi

# 添加BOM标记
#file -i $filename | grep -q "BOM"
#if [ $? -eq 1 ]; then
#    echo "add BOM to $filename"
#    sed -i '1s/^/\xef\xbb\xbf/g' $filename
#fi
sed -i '1s/^\xef\xbb\xbf//g' $filename
sed -i '1s/^/\xef\xbb\xbf/g' $filename

# 删除回车
tr -d '\r' < $filename > $filename".tmp"
mv $filename".tmp" $filename

# 替换tab为空格
sed -i 's/    /    /g' $filename

# 去除结尾多余的空格
sed -i 's/    $//g' $filename
sed -i 's/   $//g' $filename
sed -i 's/  $//g' $filename
sed -i 's/ $//g' $filename
}

for x in $*
do
    cat $x | while read myline
    do
        convert_file $myline
    done
done
