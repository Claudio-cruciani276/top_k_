KMAX=17
set -e
for i in $(ls -rL --sort=size *.hist)
do
graph="${i//@}"
FILESIZE=$(stat -Lc%s "$graph")
if [ $FILESIZE -lt 35055000 ]
then
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
python3.10 -OO k_bfs_vs_yen.py --g $graph --k $KCOUNT
python3.10 -OO k_bfs_vs_yen_with_direction.py --g $graph --k $KCOUNT
KCOUNT=$[2*$KCOUNT]
done
fi
done

