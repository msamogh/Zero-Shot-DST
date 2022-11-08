#!/bin/bash

NAME=$(randomname get)

echo "Running NegoChat-GENERATE for $NAME"

~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/escai/src/create_negochat_data.py \
    --wandb_run_id $NAME \
    --input_file ~/myblue/Zero-Shot-DST/T5DST/escai/data/negochat_corpus/negochat \
    --output_file ../data/negochat_dials.json \
    --keep_only_annotated_dials \
    --keep_only_session_ids 573391,573409,573411,575553,575571,575577 \
    --speaker_label_strategy union \
    --remove_slot_on_reject \
    --double_text_strategy merge \
    --dst_annotation_type cds \
    # --replace_with_source_ontology \
    # --only_slots area \
