#!/usr/bin/env bash

#SBATCH -J MDP_DIST # name of the project
#SBATCH -p high # duration
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 24gb
#SBATCH -C intel #request intel node (those have infiniband) # intel node
#SBATCH -o /homedtic/lsteccanella/MDP_DIST/Cluster/jobs/%N.%J.out # STDOUT # output to number of node number of job
#SBATCH -e /homedtic/lsteccanella/MDP_DIST/Cluster/jobs/%N.%j.err # STDERR # output of the error

# set -x # output verbose
source /homedtic/lsteccanella/MDP_DIST/Cluster/modules.sh
source /homedtic/lsteccanella/pytorch_env/bin/activate
python -u "/homedtic/lsteccanella/MDP_DIST/Cluster/GCSL/$@"

