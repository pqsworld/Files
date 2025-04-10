echo	"cppcheck start"
cd ..
rm doc/cppcheck_net_forward.xml
cppcheck -j 16 --language=c --addon=cert --inconclusive --xml --xml-version=2  --enable=all  . 2>doc/cppcheck_net_forward.xml
cp -f doc/cppcheck_net_forward.xml /data/guest/LIB_NET网络库/cppcheck_net_forward.xml

current_short_head=`git rev-parse --short HEAD`

echo "cppcheck end"
exit 0