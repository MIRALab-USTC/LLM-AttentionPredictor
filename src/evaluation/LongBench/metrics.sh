

# results_dir="/home/qyyang/kvcache/test/demo/runs/results_long_bench/longchat-7b-v1.5-32k_partial5_fullpredictor"
# results_dir="/home/qyyang/kvcache/test/demo/runs/results_long_bench/meta-llama-3.1-8b-instruct_quest"
results_dir='./runs/results_long_bench/longchat-7b-v1.5-32k_AttentionPredictor'

python3 "eval.py" \
    --results_dir ${results_dir} # --budget 8 
