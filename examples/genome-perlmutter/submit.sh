#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH --nodes 1 # NOTE THAT THIS NEEDS TO EQUAL CONFIG NUM_NODES
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
export MASTER_PORT=1234
export WORLD_SIZE=4 # 4 * nodes
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python ../../gene_transformer/model.py -c config.yaml # call training script