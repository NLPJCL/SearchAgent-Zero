import argparse
import json
import os
import pandas as pd
import numpy as np

DEFAULT_SYSTEM_CONTENT = """You are a helpful and harmless assistant.
Answer the given question. You must first conduct step by step reasoning between <thought> and </thought> first every time you get new information.
If you need external information, call the search tool by returning a JSON object inside <tool_call> tags.
and it will return the top searched results between <tool_response> and </tool_response>.
You can search as many times as you want. Break down the user's question into specific sub-questions for searching.
Check previous search history to ensure new queries are unique.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.
For example, <answer> Beijing </answer>. """

DEFAULT_HF_REPO_ID = "aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample"
DEFAULT_INPUT_FILENAME = "ASearcher_en_nomath_rejectsample.json"


def download_input_json(hf_repo_id, local_dir):
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {DEFAULT_INPUT_FILENAME} from {hf_repo_id} to {local_dir}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the ASearcher dataset. "
            "Please install the project requirements before running this script."
        ) from exc

    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=DEFAULT_INPUT_FILENAME,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

def process_single_item(item, split_name, index, system_content):
    # 1) question：从 extra_info.question 取
    extra = item.get("extra_info", {}) or {}
    question = extra.get("question", "")

    # 2) prompt：重建为 system+user（覆盖 placeholder）
    # user_content = user_content_prefix.rstrip("\n") + question
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    # 3) ground_truth：优先 reward_model.ground_truth，否则用 extra_info.ground_truth
    reward_model = item.get("reward_model", {}) or {}
    ground_truth = reward_model.get("ground_truth", None)
    if ground_truth is None:
        ground_truth = extra.get("ground_truth", None)

    new_ground_truth =  {"target": np.array([ground_truth])}
    # 保证 reward_model 里也带上 ground_truth（你也可以不写回去）
    reward_model_new = dict(reward_model)
    reward_model_new["ground_truth"] = new_ground_truth

    # 4) data_source：按参考代码加前缀 searchR1_
    raw_ds = extra.get("data_source", item.get("data_source", ""))
    data_source_tagged = "searchR1_" + str(raw_ds)

    # 5) tools_kwargs：对齐你参考代码
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": new_ground_truth,
                "question": question,
                "data_source": data_source_tagged,
            }
        }
    }

    # 6) extra_info：按参考代码组织；同时可保留原 extra_info 以免丢字段
    extra_info_new = {
        "index": index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split_name,
        "tools_kwargs": tools_kwargs,

        # 可选：保留原始 extra_info，方便回溯
        "raw_extra_info": extra,
    }

    return {
        "data_source": data_source_tagged,
        "prompt": prompt,
        "ability": item.get("ability", None),
        "reward_model": reward_model_new,
        "extra_info": extra_info_new,
        "metadata": item.get("metadata", None),
    }

def write_processed(processed, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".parquet"):
        df = pd.DataFrame(processed)
        df.to_parquet(output_path, index=False)
    elif output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as w:
            for x in processed:
                w.write(json.dumps(x, ensure_ascii=False) + "\n")
    else:
        raise ValueError("output_path must end with .parquet or .jsonl")

def split_train_test(data, train_ratio, seed):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    data_size = len(data)
    if data_size == 0:
        return [], []

    indices = np.arange(data_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_size = int(data_size * train_ratio)
    if data_size > 1:
        train_size = min(max(train_size, 1), data_size - 1)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo_id",
        default=DEFAULT_HF_REPO_ID,
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument(
        "--local_dir",
        default="./examples/search_agent_rl/ASearcher_en_no-math_Qwen3-8B-reject-sample",
        help="Local directory to download the raw ASearcher JSON file.",
    )
    parser.add_argument("--output_path", required=True, help="Output file path (.parquet or .jsonl)")
    parser.add_argument("--test_output_path", default=None, help="Optional test output file path (.parquet or .jsonl)")
    parser.add_argument("--split", default="train", help="Split name to write into extra_info.split when writing one output")
    parser.add_argument("--train_split", default="train", help="Split name to write into extra_info.split for train output")
    parser.add_argument("--test_split", default="test", help="Split name to write into extra_info.split for test output")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio when --test_output_path is set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    args = parser.parse_args()

    system_content = DEFAULT_SYSTEM_CONTENT
    input_json = download_input_json(args.hf_repo_id, args.local_dir)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.test_output_path:
        train_data, test_data = split_train_test(data, args.train_ratio, args.seed)
        train_processed = [
            process_single_item(item, split_name=args.train_split, index=i,
                                system_content=system_content)
            for i, item in enumerate(train_data)
        ]
        test_processed = [
            process_single_item(item, split_name=args.test_split, index=i,
                                system_content=system_content)
            for i, item in enumerate(test_data)
        ]
        write_processed(train_processed, args.output_path)
        write_processed(test_processed, args.test_output_path)
    else:
        processed = [
            process_single_item(item, split_name=args.split, index=i,
                                system_content=system_content)
            for i, item in enumerate(data)
        ]
        write_processed(processed, args.output_path)

if __name__ == "__main__":
    main()
