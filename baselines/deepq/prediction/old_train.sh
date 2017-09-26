GAME=$1
NUM_ACT=$2
COLOR=$3
TRAIN="${GAME}/train"
TEST="${GAME}/test"
MEAN="${GAME}/mean.npy"
LOG="models/${GAME}-${COLOR}-model"

python train.py --train ${TRAIN} --test ${TEST} --mean ${MEAN} --num_act ${NUM_ACT} --color ${COLOR} --log ${LOG}
