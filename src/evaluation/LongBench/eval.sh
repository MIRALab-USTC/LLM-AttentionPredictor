export CUDA_VISIBLE_DEVICES="0"

method="AttentionPredictor" # Support FullKV, tsp
sample_method="AttentionPredictor" # 
attn_implementation="flash_attention_2" # 
model_path="../../model/longchat-7b-v1.5-32k" # path to LLM model
save_dir="runs/results_long_bench" # path to result save_dir
tasks=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" "qmsum" "triviaqa" "passage_retrieval_en" "lcc" "repobench_p" "gov_report" "multi_news" "trec" "samsum" "passage_count") 
python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --sample_method ${sample_method} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --task "${tasks[*]}" \
    --skip_2layer \
    --topk 512 \
    --sink_token 64 \
    --local_token 64
# nohup bash eval.sh >> ./log/longchat_attnpred/1_4k.log 2>&1 &
python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --sample_method ${sample_method} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --task "${tasks[*]}" \
    --skip_2layer \
    --topk 1024 \
    --sink_token 64 \
    --local_token 64

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --sample_method ${sample_method} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --task "${tasks[*]}" \
    --skip_2layer \
    --topk 2048 \
    --sink_token 64 \
    --local_token 64

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --sample_method ${sample_method} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --task "${tasks[*]}" \
    --skip_2layer \
    --topk 4096 \
    --sink_token 256 \
    --local_token 256