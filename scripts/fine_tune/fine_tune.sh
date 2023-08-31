#!/bin/bash
echo "Fine tuning"

EXP_NAME=$(randomname get)

DATASET=negochat

python $SRC_DIR/train/train.py \
    --mode finetune \
    --dataset $DATASET \
    --wandb_run_id $EXP_NAME \
    --n_epochs 10 \
    --max_history 4 \
    --num_beams 5 \
    --saving_dir $CHECKPOINTS_DIR \
    --ontology_path $ONTOLOGY_DIR/$DATASET/ontology.json \
    --slot_descriptions_path $ONTOLOGY_DIR/$DATASET/slot_description.json \
    --train_batch_size 32 \
    --dev_batch_size 32 \
    --test_batch_size 32 \
    --GPU 1 \
    --slot_lang values \
    --path_train $DIALOGUES_DIR/$DATASET/train.json \
    --path_dev $DIALOGUES_DIR/$DATASET/dev.json \
    --path_test $DIALOGUES_DIR/$DATASET/test.json \
    --predictions_output_path $RESULTS_DIR/${DATASET}_results.json \
    --fix_label \
    --precision 32 \
    --keep_none_prob 0.50 \
    # --do_negative_sampling \
    # --verbose \
    # --finetune_from_ckpt $CHECKPOINTS_DIR/wary-assumption-9600.ckpt \