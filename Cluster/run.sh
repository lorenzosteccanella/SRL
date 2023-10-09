for seed in 0 1 2 3 4
do
for file in Planning/Reach_L1.py Planning/Reach_WN.py Planning/AsyReach_L1.py Planning/AsyReach_WN.py
do
    sbatch -C intel /home/lsteccanella/MAD-SemiNorm/Cluster/run-sbatch.sh "$file" "$seed"
done
done
