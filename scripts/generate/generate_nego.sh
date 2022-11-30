#!/bin/bash

NAME=$(randomname get)

echo "Running NegoChat-GENERATE for $NAME"

python $SRC_DIR/dataset/create_negochat_data.py \
    --wandb_run_id $NAME \
    --input_file $DIALOGUES_DIR/negochat/negochat \
    --output_file $DIALOGUES_DIR/negochat/negochat_dials_2.json \
    --remove_slot_on_reject \
    --double_text_strategy merge \
    --dst_annotation_type cds

