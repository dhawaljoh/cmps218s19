#!/bin/bash

# set the number of nodes
# SBATCH --mem=40GB
# set the number of tasks (processes) per node.
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1

# set name and output name of job
#SBATCH --job-name=test
#SBATCH --output ./slurm/slurm-%j.out

# set max wallclock time
#SBATCH --time=100:00:00
# mail alert at end, abortion, etc. of execution
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT_50
# send mail to this address
#SBATCH --mail-user=cchan60@ucsc.edu

# run the application
MY_HOME='/hb/home/cchan60'
source $MY_HOME/cmps218/env/bin/activate.csh
time python $MY_HOME/cmps218/cmps218s19/code/test_analogies.py -p 40 

