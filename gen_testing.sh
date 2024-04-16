KMAX=17
set -e
#for i in $(ls -rL --sort=size *.hist)
#do
graph=$1
radici=$2
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
#graph="${i//@}"
python3.10 -OO k_bfs_vs_yen_global.py --g $graph --k $KCOUNT --r $radici
python3.10 -OO k_bfs_vs_yen_with_direction.py --g $graph --k $KCOUNT --r $radici
python3.10 -OO k_bfs_vs_yen_GlobalBound.py --g $graph --k $KCOUNT --r $radici
python3.10 -OO k_bfs_vs_yen_NoPopGlobalBound.py --g $graph --k $KCOUNT --r $radici
KCOUNT=$[2*$KCOUNT]
done
#done

