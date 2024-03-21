#!/bin/bash

function sync() {

    if [[ ${TRAINER_INSTANCES_NUM} == "1" ]];
    then
        return
    else
        :
    fi

    IFS=','
    array=(${TRAINER_INSTANCES})
    for i in ${array[@]}
    do
        if [[ $i != `hostname -i` ]];then
            rsync -a --exclude '.log' --exclude '.checkpoint' ./ root@$i:`pwd`/.
        fi
    done

}

function visual() {

    ${ENV} -m tensorboard.main --logdir=./.log --port=8787 --host=`hostname`
}

function process_kill() {
    ps auxww | grep -E "(${ENV}|tensorboard)" | grep -v "notebook" | awk '{print $2}' | xargs kill -9
}


#-------------------------------------------------------------
    #   program start here!
#-------------------------------------------------------------

if [[ ${POD_INDEX} == "0" ]];
then
    sync
fi

# framework
export PADDLE_OR_TORCH="paddle"
# # of gpu per node
export CGPU_COUNT="1"
# python environment address
ENV="/opt/conda/envs/miniconda/bin/python"
# disable the __pycache__ generation
export PYTHONDONTWRITEBYTECODE=1

if [[ $1 == "" ]];
then
    visual &
    if [[ ${PADDLE_OR_TORCH} == "torch" ]];
    then
        ${ENV} -m torch.distributed.launch \
            --nproc_per_node ${CGPU_COUNT} \
            --nnodes ${TRAINER_INSTANCES_NUM} \
            --node_rank ${POD_INDEX} --master_addr ${POD_0_IP} \
            --master_port ${PADDLE_PORT} --use_env run.py  
    else
        ${ENV} -m paddle.distributed.launch \
            --master ${POD_0_IP}:${PADDLE_PORT} \
            --nnodes ${TRAINER_INSTANCES_NUM} \
            --nproc_per_node ${CGPU_COUNT} \
            --ips "${TRAINER_INSTANCES}" run.py
    fi
else
    process_kill
fi
