#!/usr/bin/env bash

#SBATCH -J MAD_sn # name of the project
#SBATCH -p high # duration
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 24gb
#SBATCH -C intel #request intel node (those have infiniband) # intel node
#SBATCH -o /home/lsteccanella/MAD-SemiNorm/Cluster/jobs_out/%N.%J.out # STDOUT # output to number of node number of job
#SBATCH -e /home/lsteccanella/MAD-SemiNorm/Cluster/jobs_err/%N.%j.err # STDERR # output of the error

# set -x # output verbose
source /home/lsteccanella/MAD-SemiNorm/Cluster/modules.sh
source /home/lsteccanella/mad_seminorm_env/bin/activate
python -u "/home/lsteccanella/MAD-SemiNorm/Cluster/$@"