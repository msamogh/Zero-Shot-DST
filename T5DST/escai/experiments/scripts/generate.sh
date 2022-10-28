#!/bin/bash

NAME=$(randomname get)

echo "Running GENERATE for $NAME"

~/myblue/woz/cai-nlp/venv/bin/python ~/myblue/Zero-Shot-DST/T5DST/escai/src/create_cai_data.py \
    --wandb_run_id $NAME \
    --input_file ~/myblue/Zero-Shot-DST/T5DST/escai/data/omnibus.xlsx \
    --output_file ../data/escai_dials.json \
    --keep_only_annotated_dials \
    --keep_only_session_ids 573391,573409,573411,575553,575571,575577 \
    --replace_escai_with_multiwoz \
    # --only_slots area \
