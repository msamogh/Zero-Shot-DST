#!/bin/bash

NAME=$(randomname get)

echo "Running NegoChat-GENERATE for $NAME"

python $SRC_DIR/dataset/negochat_generator.py \
    --negochat_root_dir $DIALOGUES_DIR/negochat/negochat \
    --output_dir $DIALOGUES_DIR/negochat \
    --double_text_strategy merge \
    --dst_annotation_type cds \
