results_dir='./runs/results_long_bench/longchat-7b-v1.5-32k_AttentionPredictor'

python3 "eval.py" \
    --results_dir ${results_dir} # --budget 8 
