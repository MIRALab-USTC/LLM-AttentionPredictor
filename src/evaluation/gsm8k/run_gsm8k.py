import re
import torch
import argparse
import jsonlines
import json
import os
import copy
import numpy as np
from tqdm import tqdm
import datasets
import time, random
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.generation import GenerationConfig
from ...llama_attention.attnpred_llama_attention import convert_kvcache_llama_attnpred

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    encoded_inputs = tokenizer(input_txt, return_tensors='pt').to(model.device)
    context_enc = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    raw_text_len = context_enc.shape[1]
    # print(f"Input text: {input_txt}\n")
    if args.top_ratio:
        layers = len(model.model.layers)
        for i in range(layers):
            if args.method == 'AttentionPredictor':
                model.model.layers[i].self_attn.topk = int(raw_text_len * args.top_ratio)
                
    outputs = model.generate(context_enc, max_new_tokens=150, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, cache_implementation="offloaded")# eos_token_id=[tokenizer.eos_token_id],
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    return output_text, raw_text_len, len(outputs[0])-raw_text_len


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def create_demo_text_from_trainset(trainset, n_shot, cot_flag=True):
    question, chain, answer = [], [], []
    for idx in range(n_shot):
        question.append(trainset[idx]["question"])
        chain.append(trainset[idx]["answer"])

    demo_text = ""
    for i in range(n_shot):
        if cot_flag:
            demo_text += f"Question: {question[i]}\nAnswer: {chain[i]}.\n\n"
        else:
            demo_text += f"Question: {question[i]}\nAnswer: .\n\n"
    return demo_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shared_data/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--fewshotnum", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="llama-3.1-8b")
    parser.add_argument("--save_dir", type=str, default="runs/")
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--top_ratio", type=float, default=None)

    args = parser.parse_args()

    set_seed(0)
    dataset = load_from_disk('../../datasets/gsm8k')

    test = dataset["test"]
    fewshot_prompt = create_demo_text_from_trainset(dataset["train"], args.fewshotnum, cot_flag=True)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    print("Loading model ...")

    model = LlamaForCausalLM.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.float16).eval()
    if args.method == 'AttentionPredictor':
        checkpoint = copy.deepcopy(model.state_dict())
        model = convert_kvcache_llama_attnpred(model, topk=args.topk, skip_2layer=True, type="AttentionPredictor", sink_token=128, local_token=128, model_type=args.model_name)
        model.load_state_dict(checkpoint, strict=False)
        print("Model converted to AttentionPredictor")

    # if args.method == 'quest':
    #     from quest_llama_attention import convert_kvcache_llama_quest
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_quest(model, topk=args.topk, skip_2layer=True)
    #     model.load_state_dict(checkpoint, strict=False)
    #     print("Model converted to QUEST")

    # elif args.method == 'h2o':
    #     from h2o_llama_attention import convert_kvcache_llama_h2o
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_h2o(model, topk=args.topk, skip_2layer=True)
    #     model.load_state_dict(checkpoint, strict=False) 
    #     print("Model converted to H2O")

    # elif args.method == 'streamingllm':
    #     from streamingllm_llama_attention import convert_kvcache_llama_streamingllm
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_streamingllm(model, topk=args.topk, skip_2layer=True)
    #     model.load_state_dict(checkpoint, strict=False)
    #     print("Model converted to StreamingLLM")  

    # elif args.method == 'snapkv':
    #     from baselines.snapkv_llama_attention import convert_kvcache_llama_snapkv
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_snapkv(model, topk=args.topk, skip_2layer=True)
    #     model.load_state_dict(checkpoint, strict=False) 
    #     print("Model converted to SnapKV") 

    model.generation_config = GenerationConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model.cuda()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log_dir = os.path.join(args.save_dir, f"{args.model_name}_{args.fewshotnum}/{args.topk}")
    if args.top_ratio:
        log_dir = os.path.join(args.save_dir, f"{args.model_name}_{args.fewshotnum}/{args.top_ratio}")
    os.makedirs(log_dir, exist_ok=True)
    
    f_output = jsonlines.Writer(open(os.path.join(log_dir, f"{args.method}.json"), "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    input_token_lens = []
    output_token_lens = []
    count = 0
    starttime = time.time()

    for doc in tqdm(test, total=tot_length, desc="Processing gsm8k"):
        context = doc_to_text(doc)
        completion, input_token_len, output_token_len = generate_sample(model, tokenizer, context)
        
        input_token_lens.append(input_token_len)
        output_token_lens.append(output_token_len)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        doc["input_token_len"] = input_token_len
        doc["output_token_len"] = output_token_len
        f_output.write(doc)
        acc_res.append(acc)
        count += 1

    f_output.close()
    print("Acc: ", np.mean(acc_res))
    print("Input Token Length - Mean: ", np.mean(input_token_lens), " Range: ", np.min(input_token_lens), "-", np.max(input_token_lens))
    print("Output Token Length - Mean: ", np.mean(output_token_lens), " Range: ", np.min(output_token_lens), "-", np.max(output_token_lens))