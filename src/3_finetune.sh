#!/bin/bash

for i in {0..34} # 0-(num_chapters-1)
do
    uv run ft.py \
        --chapters_path=hp/hp1_2_chapters.json \
        --num_epochs=5 \
        --lora_dropout=0.05 \
        --current=$i \
        --save_at_each_epoch
done

