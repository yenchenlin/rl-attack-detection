#!/bin/bash -l
#SBATCH -p long
#SBATCH -w werbos
#SBATCH --gres=gpu:k80:2
#SBATCH -J MsPacman
#SBATCH -o MsPacman.log

DATA_DIR=$1
NUM_ACT=$2
COLOR=$3
TRAIN="${DATA_DIR}/train"
TEST="${DATA_DIR}/test"
MEAN="${DATA_DIR}/mean.npy"
LOG="models/${DATA_DIR}-${COLOR}-model"

hostname
echo $CUDA_VISIBLE_DEVICES
source activate tf
export PYTHONHOME="/home/yclin/miniconda2/envs/tf"
srun python train.py --train ${TRAIN} --test ${TEST} --mean ${MEAN} --num_act ${NUM_ACT} --color ${COLOR} --log ${LOG}
