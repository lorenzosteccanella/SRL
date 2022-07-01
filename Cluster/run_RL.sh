for seed in 0 1 2 3 4
do
for discount in 0.95 0.99
do
for epsilon in 0.999 0.9999 0.99999
do
for file in OacroBot.py OcartPole.py OmountainCar.py Opendulum.py PacroBot.py PcartPole.py PmountainCar.py Ppendulum.py
do
    sbatch -C intel /homedtic/lsteccanella/MDP_DIST/Cluster/run-sbatch.sh "$file" "$seed" "$discount" "$epsilon"
done
done
done
done