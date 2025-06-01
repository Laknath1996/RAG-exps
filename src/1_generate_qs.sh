#!/bin/bash

# uv run generate_questions.py \
#     --chapters_path=hp/hp1_2_chapters.json \
#     --questions_path=hp/hp1_2_questions.json

uv run generate_questions.py \
    --chapters_path=hp/hp1_2_chapters.json \
    --questions_path=hp/hp1_2_instant_questions.json \
    --num_questions=10 \
    --scope=1 \
    --pause_iter