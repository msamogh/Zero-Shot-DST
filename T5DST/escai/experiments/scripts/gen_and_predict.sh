#!/bin/bash

NAME=$(randomname get)

# CKPT_PATH=~/myblue/Zero-Shot-DST/T5DST/save/t5-41000
CKPT_PATH=save/finetuned-t5-800
INPUT_FILE=~/myblue/Zero-Shot-DST/T5DST/escai/data/negochat_corpus/negochat
OUTPUT_FILE=../data/negochat_dials.json_test
ONTOLOGY_FILE=../ontology/negochat_ontology.json
SLOTS_DESCRIPTION_FILE=../ontology/negochat_slot_description.json
MULTIWOZ_DOMAIN=hotel
MAX_HISTORY=4

TRAIN_SPLIT=0.5
DEV_SPLIT=0.25
TEST_SPLIT=0.25

DST_ANNOTATION_TYPE=cds
DOUBLE_TEXT_STRATEGY=merge
SPEAKER_LABEL_STRATEGY=first_system
SLOT_LANG=human_collab


echo "Running NegoChat-GENERATE for $NAME"
~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/escai/src/create_negochat_data.py \
    --wandb_run_id $NAME \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --keep_only_annotated_dials \
    --speaker_label_strategy $SPEAKER_LABEL_STRATEGY \
    --remove_slot_on_reject \
    --double_text_strategy $DOUBLE_TEXT_STRATEGY \
    --dst_annotation_type $DST_ANNOTATION_TYPE \
    --data_splits $TRAIN_SPLIT,$DEV_SPLIT,$TEST_SPLIT

echo "Running PREDICT for $NAME"
~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/T5.py \
    --wandb_run_id $NAME \
    --mode predict \
    --GPU 1 \
    --ckpt_path $CKPT_PATH \
    --predict_input_file $OUTPUT_FILE \
    --predict_ontology_path $ONTOLOGY_FILE \
    --predict_slot_descriptions_path $SLOTS_DESCRIPTION_FILE \
    --predictions_output_path ../results/ \
    --speaker_label_strategy $SPEAKER_LABEL_STRATEGY \
    --slot_lang $SLOT_LANG \
    --only_domain NegoChat \
    --max_history $MAX_HISTORY \
