export CUDA_VISIBLE_DEVICES="3"

method="AttentionPredictor" # Support fullkv, AttentionPredictor
model_name="llama3.1-8b-instruct"
model_path="../../model/Llama-3-8B-Instruct-Gradient-1048k" # change to your LLM path
fewshotnum=25 # 25,47,97
topk=1024
echo "python3 run_gsm8k.py --method ${method} --fewshotnum ${fewshotnum} --topk ${topk}" 
python3 run_gsm8k.py --method ${method} --fewshotnum ${fewshotnum} --topk ${topk} #--model_path ${model_path} --model_name ${model_name}

# nohup bash run.sh >> ./log/AttentionPredictor/llama3.1_prompt4k_topk1k.log 2>&1 &
