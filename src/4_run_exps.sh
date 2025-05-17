# uv run experiment.py --results_file=results/vanilla.json
# uv run experiment.py --use_rag --results_file=results/rag.json
# uv run experiment.py --use_adapters --results_file=results/ft.json

for i in {5..1}
do
    echo "Running with checkpoint-$i"
    uv run experiment.py \
        --use_adapters \
        --results_file=results/ft_$i.json \
        --ckpt_id=$i
done

