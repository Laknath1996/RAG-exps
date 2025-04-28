#!/bin/bash

for i in {0..16}
do
    uv run ft.py --dataset_path=hp/hp1.txt --num_epochs=20 --lora_dropout=0.05 --current=$i
done

