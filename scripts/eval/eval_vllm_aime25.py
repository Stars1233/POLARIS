import os
import json
import re
import concurrent.futures
from tqdm import tqdm
from vllm import LLM, SamplingParams
import pandas as pd
# Configuration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/path/to/model")
parser.add_argument("--t", type=float, default=1.4)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--n", type=int, default=32)
parser.add_argument("--max_length", type=int, default=90000)
parser.add_argument("--experiment_name", type=str, default="Polaris-4B")
parser.add_argument("--output", type=str, default="evaluation/results")
args = parser.parse_args()

NAME = args.experiment_name
N = args.n  # number of duplications per prompt
AIME_PATH = "evaluation/benchmarks/aime25.parquet"
assert os.path.exists(AIME_PATH), f"{AIME_PATH} does not exist"
MODEL_PATH = args.model
MAX_TOKENS = args.max_length
TEMPERATURE = args.t
TOP_P = 1.0
TOP_K = args.k
OUT_PATH = f"{args.output}/{NAME}/aime25-{TEMPERATURE}-{N}-{MAX_TOKENS}-{TOP_K}.jsonl"


def load_samples(filepath):
    """Load samples and create a prompt for each sample."""
    samples = []
    df = pd.read_parquet(filepath)
    for i in range(len(df)):
        sample = {"example_id": i,"prompt": df['prompt'][i][0]['content'], "answer": df['reward_model'][i]['ground_truth']}
        samples.append(sample)
    print(f"Total samples: {len(samples)}")
    return samples


def extract_boxed_answer(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers[-1] if answers else None


def evaluate(samples):
    """Evaluate samples by comparing extracted answers with the ground truth."""
    correct = 0
    for sample in samples:
        response = sample.get("response", "")
        pred = extract_boxed_answer(response)
        if pred is not None:
            try:
                if pred and (int(pred) == int(sample["answer"])):
                    correct += 1
            except ValueError:
                pass
    accuracy = correct / len(samples)
    print(f"Pass@1: {accuracy}")
    return accuracy


def split_list(lst, n):
    """
    Split list lst into n contiguous chunks.
    Returns a list of tuples (start_idx, chunk, worker_id)
    """
    chunks = []
    total = len(lst)
    chunk_size, remainder = divmod(total, n)
    start = 0
    for worker_id in range(n):
        # Distribute the remainder one-by-one to the first few chunks.
        extra = 1 if worker_id < remainder else 0
        end = start + chunk_size + extra
        chunks.append((start, lst[start:end], worker_id))
        start = end
    return chunks


def worker_process(args):
    """
    Worker process to run vLLM inference on a chunk of samples.
    Each worker sets CUDA_VISIBLE_DEVICES to the assigned GPU.
    """
    start_idx, chunk, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker on GPU {gpu_id}] Processing {len(chunk)} samples...")

    llm = LLM(model=MODEL_PATH, enforce_eager=True)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS)
    messages = [
        [{"role": "user", "content": sample["prompt"]}]
        for sample in chunk
    ]
    responses = llm.chat(messages, sampling_params, use_tqdm=True)
    responses = [r.outputs[0].text for r in responses]
    return start_idx, responses


def main():
    # 1. Load samples and duplicate them K times.
    samples = load_samples(AIME_PATH)
    samples = samples * N
    total_samples = len(samples)
    print(f"Total samples after duplication: {total_samples}")

    # 2. Split samples into 8 chunks (one per GPU/worker).
    num_workers = 8
    chunks = split_list(samples, num_workers)

    # 3. Launch 8 worker processes to run inference via vLLM.
    results = [None] * total_samples
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_process, args) for args in chunks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
            start_idx, responses = future.result()
            results[start_idx:start_idx + len(responses)] = responses

    print(f"Total responses collected: {len(results)}")

    # 4. Attach responses to the corresponding samples.
    for sample, response in zip(samples, results):
        sample["response"] = response

    # 5. Save the outputs.
    with open(OUT_PATH, 'w') as out_file:
        for sample in samples:
            out_file.write(json.dumps(sample) + "\n")
    print(f"Saved results to {OUT_PATH}")

    # 6. Evaluate accuracy.
    with open(OUT_PATH, 'r') as f:
        samples = [json.loads(line) for line in f]
    evaluate(samples)


if __name__ == "__main__":
    main()
