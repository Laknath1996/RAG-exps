#!/bin/bash

for i in {0..15}
do
    uv run ft.py --current=$i
done

