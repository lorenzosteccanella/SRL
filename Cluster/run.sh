for seed in 0 1 2 3 4
do
for file in Pendulum_PER.py
do
    sbatch -C intel /homedtic/lsteccanella/MDP_DIST/Cluster/run-sbatch.sh "$file" "$seed"
done
done
