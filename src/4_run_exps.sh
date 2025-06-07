# uv run experiment.py --results_file=results/vanilla.json
# uv run experiment.py --use_rag --results_file=results/rag.json
# uv run experiment.py --use_adapters --results_file=results/ft.json

# for i in {5..1}
# do
#     echo "Running with checkpoint-$i"
#     uv run experiment.py \
#         --use_adapters \
#         --results_file=results/ft_$i.json \
#         --ckpt_id=$i
# done

QUESTIONS_PATH="hp/hp1_2_instant_questions.json"

# uv run experiment.py \
#     --questions_path=$QUESTIONS_PATH \
#     --results_file=results/vanilla.json \

# uv run experiment.py \
#     --use_rag \
#     --questions_path=$QUESTIONS_PATH \
#     --results_file=results/rag.json

# uv run experiment.py \
#     --use_adapters \
#     --ckpt_id=1 \
#     --questions_path=$QUESTIONS_PATH \
#     --results_file=results/ft_1.json

uv run experiment.py \
    --use_rag \
    --rag_memory=5 \
    --questions_path=$QUESTIONS_PATH \
    --results_file=results/rag_limited_5.json

uv run experiment.py \
    --use_rag \
    --rag_memory=-1 \
    --questions_path=$QUESTIONS_PATH \
    --results_file=results/rag_unlimited.json