#!/bin/bash
for name in USAir NS Power Celegans Router PB Ecoli Yeast
do
   for i in 1 2 3 4 5 6 7 8 9 10
   do
      sbatch ./dash/wo_attr.sh $i $i $name
   done
done

for name in cora citeseer pubmed
do
   for feature in vgae gic argva
   do
      for i in 1 2 3 4 5 6 7 8 9 10
      do
         sbatch ./dash/w_attr.sh $i $i $name $feature
      done
   done
done