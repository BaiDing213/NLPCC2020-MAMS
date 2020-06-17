#!/bin/bash

R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=./ernie:${PYTHONPATH:-}

if [[ -f ./model_conf ]];then
    source ./model_conf
else
    export CUDA_VISIBLE_DEVICES=0
fi

CURDIR=$(cd "$(dirname "$0")";pwd)
ERNIEROOT=$(cd "$(dirname $CURDIR)";pwd)

TASK=ATSA
SCALE=base
if [[ $TASK = 'ATSA' ]] || [[ $TASK = 'ACSA' ]] ; then
    DATAROOT=$ERNIEROOT/data/$TASK
else
    echo "WRONG TASKNAME."
    exit 1
fi
if [[ $SCALE = 'base' ]] || [[ $SCALE = 'large' ]]; then
    MODELROOT=$ERNIEROOT/pretrained-models/ERNIE_2_en_$SCALE
else
    echo "WRONG TASKNAME."
    exit 1
fi
CHECKPOINT=$ERNIEROOT/saved-checkpoints/$TASK/$SCALE
mkdir -p log/

for i in {0..0};do

    timestamp=`date "+%Y-%m-%d-%H-%M-%S"`

    python -u $ERNIEROOT/ernie/run_classifier.py                                                \
               --use_cuda True                                                 \
               --for_cn False                                                  \
               --use_fast_executor ${e_executor:-"true"}                       \
               --tokenizer ${TOKENIZER:-"FullTokenizer"}                       \
               --use_fp16 ${USE_FP16:-"false"}                                 \
               --do_train true                                                 \
               --do_val true                                                   \
               --do_test False                                                  \
               --batch_size 4                                                  \
               --init_pretraining_params $MODELROOT/params                  \
               --ernie_config_path $MODELROOT/ernie_config.json             \
               --vocab_path $MODELROOT/vocab.txt                            \
               --verbose true                                                  \
               --train_set $DATAROOT/train.tsv                     \
               --dev_set   $DATAROOT/dev.tsv                       \
               --test_set  $DATAROOT/test.tsv                      \
               --checkpoints $CHECKPOINT                                   \
               --save_steps 250                                               \
               --weight_decay  0.0                                             \
               --warmup_proportion 0.1                                         \
               --validation_steps 250                                  \
               --epoch 10                                                       \
               --max_seq_len 128                                               \
			   --metric acc_and_f1											   \
               --learning_rate 2e-5                                            \
               --skip_steps 10                                                 \
               --num_iteration_per_drop_scope 1                                \
               --num_labels 3                                                  \
               --for_cn  False                                                 \
               --test_save output/test_out.$i.tsv                              \
               --random_seed 1 2>&1 | tee  log/job.$i.$timestamp.log           \

done