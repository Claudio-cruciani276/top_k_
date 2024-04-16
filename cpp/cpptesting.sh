make release
KMAX=256
set -e
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
./main -g critical.hist -k $KCOUNT -o 1 -q 100000 -d 0
KCOUNT=$[2*$KCOUNT]
done
make valgrind && valgrind --tool=callgrind  ./main -g i0i.bgr.hist -k 2 -q 10 -o 1 -d 0
make release
KMAX=256
set -e
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
for i in $(ls -rL --sort=size *.hist)
do
graph="${i//@}"
./main -g $graph -k $KCOUNT -o 1 -q 100000 -d 0
done
KCOUNT=$[2*$KCOUNT]
done

