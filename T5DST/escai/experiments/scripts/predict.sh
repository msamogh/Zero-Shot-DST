#!/bin/bash

NAME=$1

echo "Running PREDICT for $NAME"

~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/T5.py \
    --wandb_run_id $NAME \
    --mode predict \
    --GPU 1 \
    --ckpt_path ~/myblue/Zero-Shot-DST/T5DST/save/t5-41000 \
    --predict_input_file ../data/negochat_dials.json \
    --predict_ontology_path ../ontology/negochat_ontology.json \
    --predict_slot_descriptions_path ../ontology/negochat_slot_description.json \
    --predictions_output_path ../results/ \
    --speaker_label_strategy union \
    \
    --slot_lang human \
    --only_domain NegoChat \
    # --perform_multiwoz_to_escai