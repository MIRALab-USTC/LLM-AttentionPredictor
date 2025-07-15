import os
import json
import random
import argparse
import copy

import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from ...llama_attention.attnpred_llama_attention import convert_kvcache_llama_attnpred
# from ...llama_attention.h2o_llama_attention import convert_kvcache_llama_h2o
# from ...llama_attention.quest_llama_attention import convert_kvcache_llama_quest
# from ...llama_attention.streamingllm_llama_attention import convert_kvcache_llama_streamingllm
# from ...llama_attention.snapkv_llama_attention import convert_kvcache_llama_snapkv


dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench_p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench_p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama-3.1": 1047500, 
    "llama-3": 1047500,
    "mistral": 31500,
    "longchat": 31500,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(tokenizer, prompt, model_name):
    if "longchat" in model_name: 
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or "llama-2" in model_name or "lwm" in model_name:
        print('llama2', model_name)
        prompt = f"[INST]{prompt}[/INST]"
    elif "llama-3" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    model_name = model_path.split("/")[-1]
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "repobench_p", "lcc"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
                
            example["prompt"] = prompt
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
    print(f"QA Num is {len(test_data)}")
        
    
    
    for example in test_data:
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    avg_length = sum(lengths)/len(lengths)
    print(f"longchat dataset: {args.dataset}, avg_word_length: {avg_length}")
    
    model_name = model_path.split("/")[-1]

    save_dir = os.path.join(args.save_dir, f"{model_name}_{args.sample_method}/{args.topk}", args.dataset)

    if args.sample_method == "sensitivity_history_step":
        save_dir = os.path.join(args.save_dir, f"{model_name}_{args.sample_method}/{args.history_step}", args.dataset)
    elif args.sample_method == "sensitivity_calibration":
        save_dir = os.path.join(args.save_dir, f"{model_name}_{args.sample_method}/{args.calibration_step}", args.dataset)
    elif args.sample_method == "sensitivity_block_size":
        save_dir = os.path.join(args.save_dir, f"{model_name}_{args.sample_method}/{args.block_size}", args.dataset)

    os.makedirs(save_dir, exist_ok=True)

    fout = open(os.path.join(save_dir, f"{args.method}.json"), "w")
    token_length = []
    output_len = []
    skip = 0
     
    for i in tqdm(range(0, len(prompts))):
        
        prompt = prompts[i]
        input = inputs[i]
        context = contexts[i]
        answers = answerss[i]
        length = lengths[i]
        
        dataset = datasets[i]
        language = languages[i]
        all_classes = all_classess[i]
        _id = _ids[i]
        
        tokenized_prompts = tokenizer([prompt], return_tensors="pt").to('cuda')
        input_ids = tokenized_prompts.input_ids
        if len(input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(input_ids[0][-half:], skip_special_tokens=True)
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            input_ids = tokenized_prompts.input_ids
        
        context_length = input_ids.shape[-1]
        token_length.append(context_length)


        try:
            if dataset == 'samsum' or 'llama-3' in args.model_path.lower():
                output = model.generate(
                    **tokenized_prompts,
                    output_attentions = False,
                    max_new_tokens=output_max_len,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )
            else: 
                output = model.generate(
                    **tokenized_prompts,
                    output_attentions = False,
                    max_new_tokens=output_max_len,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id]
                )


            outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)[0]
            output_len.append(output[0][context_length:].shape[-1])
            

            if "llama-3" in model_path.lower():
                outputs = outputs.split("<\s>")[0]
        
            torch.cuda.empty_cache()
                
            example = {}
            # example["prompt"] = prompt
            # example["input"] = input
            # example["context"] = context
            example["answers"] = answers
            example["pred"] = outputs
            example["length"] = length
            example["input_tokens"] = context_length
            example["dataset"] = dataset
            example["language"] = language
            example["all_classes"] = all_classes
            example["_id"] = _id

            fout.write(json.dumps(example) + "\n")
        except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"CUDA out of memery. Skip item {i} with length {context_length}")
                    skip += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
    print(f"{skip} items have been skipped.")
    
    avg_token_length = sum(token_length)/len(token_length)
    avg_output_length = sum(output_len)/len(output_len)
    print(f"longchat dataset: {args.dataset}, avg_token_length: {avg_token_length}, avg_output_length: {avg_output_length}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="log")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    
    parser.add_argument("--sample_method", type=str, default="FullKV")
    
    parser.add_argument("--attn_implementation", type=str,  default="eager", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--task", type=str, default=None, help="dataset to evaluate on.")
    parser.add_argument("--skip_2layer", action='store_true', help="skip the bottom 2 layers of the model.")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--sink_token", type=int, default=64)
    parser.add_argument("--local_token", type=int, default=64)
    parser.add_argument("--history_step", type=int, default=64)
    parser.add_argument("--calibration_step", type=int, default=5)
    parser.add_argument("--block_size", type=int, default=16)

    
    args = parser.parse_args()
    set_seed(args.seed)

    datasets = args.task.split()
    print(f"datasets: {datasets}")
    print(f'topk: {args.topk}')
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    
    model = LlamaForCausalLM.from_pretrained(args.model_path, attn_implementation=args.attn_implementation, device_map="auto", torch_dtype=torch.float16)
    if args.method == 'AttentionPredictor':
        checkpoint = copy.deepcopy(model.state_dict())
        model = convert_kvcache_llama_attnpred(model, model_type=args.model_name, topk=args.topk, skip_2layer=args.skip_2layer, type=args.sample_method, sink_token=args.sink_token, local_token=args.local_token, calibration_step=args.calibration_step)
        model.load_state_dict(checkpoint, strict=False)

        # if args.sample_method == "sensitivity_history_step":
        #     for i in range(32):
        #         model.model.layers[i].self_attn.history_step = args.history_step
        # if args.sample_method == "sensitivity_calibration":
        #     for i in range(32):
        #         model.model.layers[i].self_attn.calibration_interval = args.calibration_step
        # if args.sample_method == "sensitivity_block_size":
        #     for i in range(32):
        #         model.model.layers[i].self_attn.pooling_block_size = args.block_size

    # elif args.method == 'h2o':
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_h2o(model, topk=args.topk, skip_2layer=args.skip_2layer)
    #     model.load_state_dict(checkpoint, strict=False)    

    # elif args.method == 'quest':
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_quest(model, topk=args.topk, skip_2layer=args.skip_2layer, chunk_size=16)
    #     model.load_state_dict(checkpoint, strict=False)  

    # elif args.method == 'streamingllm':
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_streamingllm(model, topk=args.topk, skip_2layer=args.skip_2layer)
    #     model.load_state_dict(checkpoint, strict=False)  

    # elif args.method == 'snapkv':
    #     checkpoint = copy.deepcopy(model.state_dict())
    #     model = convert_kvcache_llama_snapkv(model, topk=args.topk, skip_2layer=args.skip_2layer)
    #     model.load_state_dict(checkpoint, strict=False)        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'llama-3' in args.model_path.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        
    model.eval().cuda()
    
    save_dir = args.save_dir

    start_time = datetime.now()
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on  dataset {dataset} - {idx}/{len(datasets)}")
        
        args.dataset = dataset
        
        args.data_file = f"../../dataset/LongBench/{args.dataset}.jsonl"
        
        main(args)
        duration = datetime.now() - start_time
        print(f"time after start is {str(timedelta(seconds=duration.total_seconds()))}.")
