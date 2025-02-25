import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench_p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--budget', type=int, default=1024)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        scores.append(score)
    return round(100 * total_score / len(predictions), 2), scores

if __name__ == '__main__':
    args = parse_args()

    # budget = [args.budget]
    budget = [512,1024,2048,4096]
    
    dataset_list = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench_p"
        ]
    
    
    results_dir = args.results_dir
    for i in budget:
        results_list = [
            ["dataset"],
            # ["FullKV"],
            ["AttentionPredictor"],
            # ["h2o"],
            # ["quest"],
            # ["streamingllm"],
            # ["snapkv"],
        ]
        args.results_dir = os.path.join(results_dir, f"{i}")

        for dataset in dataset_list:
            
            results_list[0].append(dataset)
            predictions_full = []
            predictions_tsp = []
            score_full = 0
            score_tsp = 0
            
            for idx, method in enumerate(["AttentionPredictor"]):
                try:
                    args.method = method
                    args.dataset = dataset
                    args.eval_file = os.path.join(args.results_dir, dataset,f"{method}.json")
                    

                    scores = dict()
                    predictions, answers, lengths = [], [], []
                    with open(args.eval_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                predictions.append(data["pred"])
                                answers.append(data["answers"])
                                all_classes = data["all_classes"]
                                if "length" in data:
                                    lengths.append(data["length"])
                            except:
                                print("error")
                    if args.longbench_e:
                        score = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    else:
                        score, scores1 = scorer(args.dataset, predictions, answers, all_classes)
                        if args.dataset == 'qasper':
                            score_e = scorer_e(args.dataset, predictions, answers, lengths, all_classes)
                    scores[args.dataset] = score
                        
                    output_dir = os.path.dirname(args.eval_file)
                    results_list[idx+1].append(score)
                    
                    with open(os.path.join(output_dir, f"metrics_{args.method}.json"), "w") as f:
                        json.dump(scores, f, ensure_ascii=False, indent=4)
                
                    print(f"dataset {args.dataset} method {args.method} scores {scores}")
                except:
                    
                    results_list[idx+1].append(-1)
                    print(f"dataset {args.dataset} method {args.method} scores {None}")

                    
        import csv
        with open(os.path.join(args.results_dir,f"results.csv"), 'a') as fp:
            writer = csv.writer(fp)
            writer.writerows(results_list)
