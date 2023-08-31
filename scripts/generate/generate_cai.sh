#!/bin/bash

NAME=$(randomname get)

echo "Running ESCAI-GENERATE for $NAME"

python $SRC_DIR/dataset/cai_generator.py \
    --excel_file_path $DIALOGUES_DIR/escai/omnibus.xlsx \
    --output_dir $DIALOGUES_DIR/escai \
    --keep_only_annotated_dials \
    --keep_only_session_ids 573391,573409,573411,575553,575571,575577 \
    --double_text_strategy merge \
    --dst_annotation_type cds \
    # --remove_slot_on_reject \
