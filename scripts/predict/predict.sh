#!/bin/bash

EXP_NAME=$(randomname get)
FINE_TUNE_FROM_EXP=$1
# FINE_TUNE_FROM_STEP=$2

DATASET=negochat

echo "Running PREDICT for $EXP_NAME (from $FINE_TUNE_FROM_EXP)"

python $SRC_DIR/train/train.py \
    --wandb_run_id $EXP_NAME \
    --mode predict \
    --GPU 5 \
    --hf_checkpoint $CHECKPOINTS_DIR/$FINE_TUNE_FROM_EXP \
    --dataset $DATASET \
    --path_predict $DIALOGUES_DIR/$DATASET/test.json \
    --ontology_path $ONTOLOGY_DIR/$DATASET/ontology.json \
    --slot_descriptions_path $ONTOLOGY_DIR/$DATASET/slot_description.json \
    --predictions_output_path $RESULTS_DIR/$EXP_NAME \
    --slot_lang values \
    --max_history 2 \
    --predict_batch_size 64 \
    --verbose
    # --num_beams 5 \
    # --do_sample \
    # --top_k 20 \
    # --predict
