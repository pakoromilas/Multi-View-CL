#!/bin/sh
#SBATCH -t 3:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/Simclr/HPS/cifar10/bsz64/sgd_resnet18/02.out
#SBATCH --job-name=Simclr_HPS_cifar10_bsz64_sgd_resnet18_02
#SBATCH --mail-user=pakoromilas@di.uoa.gr
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

echo "--- GPU allocated for $1 ."

echo "Starting at `date`"
echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_NNODES nodes."i
echo "Job name: $SLURM_JOB_NAME"
echo "Job id: $SLURM_JOBID"

host=$(hostname -I)
host=${host#* }
host=${host% *}
export host

#module load AutoDock-GPU/1.5.3-GCC-10.3.0-CUDA-11.3.1
#module load CUDA/11.7.0
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
#module load Python/3.9.5-GCCcore-10.3.0
#module load Python/3.10.8-GCCcore-12.2.0-bare
#module load Anaconda3/2021.05
#module load Anaconda3/2021.11

srun bash run_train_supcon.sh configs/baseline/cifar10/bsz64/sgd/cifar_train_epochs200_bs64_02.yaml NTXentHPS 1234