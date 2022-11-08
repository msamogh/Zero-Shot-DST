#!/bin/bash

rm -rf t5-small/

NAME=$(randomname get)

CKPT_PATH=~/myblue/Zero-Shot-DST/T5DST/save/t5-41000
# CKPT_PATH=save/finetuned-t5-0
INPUT_FILE=~/myblue/Zero-Shot-DST/T5DST/escai/data/negochat_corpus/negochat
OUTPUT_FILE=../data/negochat_dials.json
TRAIN_FILE=../data/negochat_dials.json_train
VAL_FILE=../data/negochat_dials.json_dev
TEST_FILE=../data/negochat_dials.json_test
ONTOLOGY_FILE=../ontology/negochat_ontology.json
SLOTS_DESCRIPTION_FILE=../ontology/negochat_slot_description.json
MULTIWOZ_DOMAIN=hotel
MAX_HISTORY=4

DEV_AND_TEST_SPLIT=0.90
TEST_SPLIT=0.50``

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
    --data_splits $DEV_AND_TEST_SPLIT,$TEST_SPLIT

echo "Running FINE_TUNE for $NAME"
~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/T5.py \
    --train_batch_size 128 \
    --dev_batch_size 128 \
    --wandb_run_id $NAME \
    --mode finetune \
    --GPU 2 \
    --ckpt_path $CKPT_PATH \
    --predict_input_file $TRAIN_FILE \
    --predict_ontology_path $ONTOLOGY_FILE \
    --predict_slot_descriptions_path $SLOTS_DESCRIPTION_FILE \
    --predictions_output_path ../results/ \
    --speaker_label_strategy $SPEAKER_LABEL_STRATEGY \
    --slot_lang $SLOT_LANG \
    --only_domain NegoChat \
    --max_history $MAX_HISTORY \
    --path_train $TRAIN_FILE \
    --path_dev $VAL_FILE \
    --path_test $TEST_FILE
