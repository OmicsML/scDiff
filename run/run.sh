#!/bin/bash --login
# Uncomment specific block of interest and run
# $ bash script.sh
#
# Specify CUDA devices
# $ CUDA_VISIBLE_DEVICES=1 bash script.sh
#
# Specify log directory
# $ LOGDIR=/localscrtach/scdiff bash script.sh
#
# To view the generated script without executing, pass the TEST_FLAG envar as 1
# $ TEST_FLAG=1 bash script.sh

trap "echo ERROR && exit 1" ERR

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
NAME=v7.5

CONFIG_PREFIX="configs/eval"

TEST_FLAG=${TEST_FLAG:-0}

# OFFLINE_SETTINGS="--wandb f"
OFFLINE_SETTINGS="--wandb_offline t"
# --------------------

HOMEDIR=$(dirname $(dirname $(realpath $0)))
cd $HOMEDIR
echo HOMEDIR=$HOMEDIR

launch () {
    full_settings=($@)
    task=${full_settings[0]}
    seed=${full_settings[1]}

    if [[ $task == denoising ]] || [[ $task == perturbation ]]; then
        if [[ $task == denoising ]]; then
            dataset_name=PBMC1K
        elif [[ $task == perturbation ]]; then
            dataset_name=salmonella
        fi

        if [[ $task == denoising ]]; then
            data_prefix="_10"
        else
            data_prefix=""
        fi
        data_settings="data.params.train.params.dataset=${dataset_name} data.params.train.params.fname=${dataset_name}${data_prefix}_processed.h5ad"
        data_settings+=" data.params.test.params.dataset=${dataset_name} data.params.test.params.fname=${dataset_name}${data_prefix}_processed.h5ad"

    elif [[ $task == genepert ]]; then
        dataset_name=norman
        task_specific_settings+=" lightning.trainer.max_epochs=100"
        data_settings="data.params.train.params.dataset=${dataset_name}  data.params.validation.params.dataset=${dataset_name} data.params.test.params.dataset=${dataset_name}"

    elif [[ $task == annotation ]]; then
        name=pbmc12k
        dataset_name=PBMC12K
        data_settings="data.params.train.target=scdiff.data.${name}.${dataset_name}Train"
        data_settings+=" data.params.train.params.fname=${dataset_name}_processed.h5ad"
        data_settings+=" data.params.test.target=scdiff.data.${name}.${dataset_name}Test"
        data_settings+=" data.params.test.params.fname=${dataset_name}_processed.h5ad"

    else
        echo Unknown task $task && exit 1
    fi

    script="python main.py -b ${CONFIG_PREFIX}_${task}.yaml --name ${NAME} --seed ${seed}"
    [[ $task == annotation_cellxgene ]] && task=annotation
    script+=" --logdir ${LOGDIR} --postfix ${task}_${name}_r.${seed} ${OFFLINE_SETTINGS} ${data_settings} ${task_specific_settings}"

    echo task=$task dataset_name=$dataset_name seed=$seed
    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


task=$1
SEED=${SEED:-10}

launch ${task} ${SEED}
