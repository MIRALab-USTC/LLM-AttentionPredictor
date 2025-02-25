export CUDA_VISIBLE_DEVICES="0"

method="AttentionPredictor" # Support FullKV, tsp
sample_method="AttentionPredictor" # 1k,2k,4k,all,skip_2layer, skip_2_layer_true, pool
attn_implementation="flash_attention_2" # Support "flash_attention_2", "eager".
# model_path="/home/qyyang/kvcache/model/Llama-3-8B-Instruct-Gradient-1048k"
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
# nohup bash eval_partial.sh >> ./log/longchat_tsp_partial_full_predictor/1_4k_sink.log 2>&1 &
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