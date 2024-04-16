#echo $1

KMAX=256
set -e
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
python3.10 -OO yens_based_ksisp.py --g critical.hist --k $KCOUNT
#python3 -OO yens_based_ksisp.py --g critical.hist --k $KCOUNT --b 1
KCOUNT=$[2*$KCOUNT]
done

KMAX=256
set -e
KCOUNT=2
while [ $KCOUNT -lt $KMAX ]
do
for i in $(ls -rL --sort=size *.hist)
do
graph="${i//@}"
python3.10 -OO yens_based_ksisp.py --g $graph --k $KCOUNT 
#python3 -OO yens_based_ksisp.py --g $graph --k $KCOUNT --b 1
done
KCOUNT=$[2*$KCOUNT]
done

