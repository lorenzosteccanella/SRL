for seed in 0 1 2 3 4
do
for file in AcroBot.py Acrobot_wo_h.py CartPole.py CartPole_wo_h.py MountainCar.py MountainCar_wo_h.py Pendulum.py Pendulum_wo_h.py PointMass.py PointMass_wo_h.py
do
    sbatch -C intel /homedtic/lsteccanella/MDP_DIST/Cluster/run-sbatch.sh "$file" "$seed"
done
done
