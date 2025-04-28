#!/bin/bash

uv run generate_embeddings.py \
    --book_path=hp/hp2.txt \
    --database_path=hp_vdbs \
    --collection_name=book2