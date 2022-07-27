import subprocess
from argparse import ArgumentParser
from pathlib import Path

template_string = """#!/bin/bash
#SBATCH -A m3957_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t {}
#SBATCH --nodes {}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --output={}.out
#SBATCH --error={}.err
export MASTER_PORT=1234
export WORLD_SIZE={}
echo "NODELIST="${{SLURM_NODELIST}}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load pytorch/1.9
cd /global/cfs/cdirs/m3957/mzvyagin/gene_transformer/gene_transformer/
srun python model.py -c {}
"""


def format_and_submit(args, num_gpus_per_node=4):
    """Add arguments to an sbatch script, save to temp file, and submit to slurm scheduler"""
    total_gpus = int(args.nodes) * 4
    slurm_output_filename = args.config.with_suffix("").name
    new_script = template_string.format(args.time, args.nodes, slurm_output_filename, slurm_output_filename, total_gpus, args.config)
    with open("/tmp/gene_transformer_script.sbatch", "w") as f:
        f.write(new_script)
    # TODO: add option to add job name
    subprocess.run("sbatch /tmp/gene_transformer_script.sbatch", shell=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--time", default="4:00:00")
    parser.add_argument("-n", "--nodes", default="4")
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()
    format_and_submit(args)
