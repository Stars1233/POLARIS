import sys

from utils import grade_answer_verl

from transformers import AutoTokenizer


def get_len(seq):
    return len(tokenizer.encode(seq))


def get_diverse_score(sequences, n=4):
    """
    计算给定多个序列的 Distinct-n 分数。

    sequences: List[str] 要评估的序列列表
    n: int, n-gram 阶数，默认为 1
    """
    distinct_ngrams = set()
    total_ngrams = 0

    for seq in sequences:
        # more accurate n-gram
        # tokens = nltk.word_tokenize(seq)
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1

    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0


import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, help='Path to load the dataset jsonl files', default="")
parser.add_argument("--calc_length", action="store_true")
args = parser.parse_args()
file_name = args.file_name
if args.calc_length:
    tokenizer = AutoTokenizer.from_pretrained("/path/to/the/model")
else:
    tokenizer = None

diverse = []
avg_scores = []
avg_scores2 = []

df = pd.read_parquet(file_name)
best = []
bad_samples = 0
solve_none = 0
solve_all = 0
without_boxed = 0
response_lengths = []
print("Total data:", len(df), "Number of rollouts:", len(df["responses"][0]))
for i in range(len(df)):
    responses = df["responses"][i]
    gt = df["reward_model"][i]["ground_truth"]
    responses_list = [str(response) for response in responses]
    if tokenizer:
        response_lengths += [get_len(response) for response in responses_list]
    else:
        response_lengths = [0]
    not_formated = ["boxed" not in response for response in responses_list]
    without_boxed += sum(not_formated)
    scores = [grade_answer_verl(response, gt) for response in responses_list]
    diverse.append(get_diverse_score(responses_list))
    avg_score = sum(scores) / len(scores)
    avg_scores.append(avg_score)
    best.append(max(scores))

    if avg_score == 0:
        solve_none += 1
    elif avg_score == 1:
        solve_all += 1

print("============ performance ===============")
print("avg scores: ", sum(avg_scores) / len(avg_scores))
print("============ other info ================")
print("best score: ", sum(best) / len(best))
print("solve none:", solve_none)
print("solve all: ", solve_all)
print("distinct 4-gram: ", sum(diverse) / len(diverse))
print("avg output length: ", sum(response_lengths) / len(response_lengths))
print("format error rollouts: ", without_boxed / 960)
