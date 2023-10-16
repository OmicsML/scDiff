#!/bin/bash --login

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOGDIR=${LOGDIR:-logs}
N_PRESERVE=1  # one-shot

# OFFLINE_SETTINGS="--wandb f"
OFFLINE_SETTINGS="--wandb_offline t"
# --------------------

HOMEDIR=$(dirname $(dirname $(realpath $0)))
cd $HOMEDIR
echo LOGDIR=$LOGDIR
echo HOMEDIR=$HOMEDIR

TOPK=${TOPK:-3}  # 2-8
SEED=${SEED:-10}

echo topk=${TOPK}, seed=${SEED}

VERSION=fs_v2.0
OPTION="-w t -lc f"
BASE_CONFIG="model.params.recon_flag=false \
data.params.train.params.n_preserve=${N_PRESERVE} \
data.params.train.params.choice=top \
data.params.test.params.n_preserve=${N_PRESERVE} \
data.params.test.params.choice=top"
TOPK_CONFIG="data.params.train.params.num_cell_types=${TOPK} data.params.test.params.num_cell_types=${TOPK}"
CONFIG_FILE=configs/cellxgene_topk_finetune.yaml
dataset=HLCA_sub
CLASSIFIER_CONFIG="model.params.classifier_config.params.conds_to_fix=batch"

# LLM conditioner (BioLinkBert)
PREFFIX="cl"
THRESHOLD=1000
PRETRAINED_CKPT_PATH="biolinkbert_HLCA_sub_threshold1000.ckpt"
ADDITIONAL_CONFIG="data.params.train.params.text_cond_flag=true \
    data.params.train.params.text_null_flag=true \
    data.params.test.params.text_cond_flag=true \
    data.params.test.params.text_null_flag=true \
    data.params.train.params.dataset=${dataset} \
    data.params.test.params.dataset=${dataset} \
    model.params.model_config.params.text_emb_file=${dataset}-cl-emb.pt \
    data.params.train.params.threshold=${THRESHOLD} \
    data.params.test.params.threshold=${THRESHOLD}"
PRETRAINED_CONFIG="pretrained_ckpt_path=data/${PRETRAINED_CKPT_PATH}"

NAME="${VERSION}_${PREFFIX}_${dataset}_n${N_PRESERVE}_top${TOPK}_EPOCHS}_finetune_r.${SEED}"
BASE_SCRIPt="python main.py -b ${CONFIG_FILE} ${OFFLINE_SETTINGS} --logdir ${LOGDIR} --seed ${SEED} ${OPTION}"
script="${BASE_SCRIPt} --postfix ${NAME} ${BASE_CONFIG} ${TOPK_CONFIG} ${PRETRAINED_CONFIG} ${ADDITIONAL_CONFIG}"
echo ${script} && eval ${script}
