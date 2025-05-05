#!/bin/bash

uv run generate_embeddings.py \
    --chapters_path=hp/hp1_2_chapters.json \
    --database_path=hp_vdbs/hp \
    --collection_name=book1_2