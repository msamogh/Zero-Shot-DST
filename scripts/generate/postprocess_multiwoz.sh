#!/bin/bash

python \
    $SRC_DIR/dataset/postprocess_multiwoz.py \
    --input_path $DIALOGUES_DIR/multiwoz/dev_dials.json \
    --output_path $DIALOGUES_DIR/multiwoz/dev_dials_processed.json \
    --affirm_single_slot_verbalizer_path $VERBALIZERS_DIR/affirm_single.txt \
    --affirm_multiple_slots_verbalizer_path $VERBALIZERS_DIR/affirm_multiple.txt

python \
    $SRC_DIR/dataset/postprocess_multiwoz.py \
    --input_path $DIALOGUES_DIR/multiwoz/train_dials.json \
    --output_path $DIALOGUES_DIR/multiwoz/train_dials_processed.json \
    --affirm_single_slot_verbalizer_path $VERBALIZERS_DIR/affirm_single.txt \
    --affirm_multiple_slots_verbalizer_path $VERBALIZERS_DIR/affirm_multiple.txt

python \
    $SRC_DIR/dataset/postprocess_multiwoz.py \
    --input_path $DIALOGUES_DIR/multiwoz/test_dials.json \
    --output_path $DIALOGUES_DIR/multiwoz/test_dials_processed.json \
    --affirm_single_slot_verbalizer_path $VERBALIZERS_DIR/affirm_single.txt \
    --affirm_multiple_slots_verbalizer_path $VERBALIZERS_DIR/affirm_multiple.txt