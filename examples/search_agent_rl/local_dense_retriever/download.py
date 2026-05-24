# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/download.py


import argparse
import os

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument(
    "--repo_id",
    type=str,
    default=None,
    help="Deprecated alias for --index_repo_id.",
)
parser.add_argument(
    "--index_repo_id",
    type=str,
    default="PeterJinGo/wiki-18-e5-index",
    help="Hugging Face dataset repository ID for the dense index shards.",
)
parser.add_argument(
    "--corpus_repo_id",
    type=str,
    default="PeterJinGo/wiki-18-corpus",
    help="Hugging Face dataset repository ID for the retrieval corpus.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Local directory to save files. Defaults to ./search_data beside this script.",
)

args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.abspath(args.save_path or os.path.join(script_dir, "search_data"))
os.makedirs(save_path, exist_ok=True)
index_repo_id = args.repo_id or args.index_repo_id

for file in ["part_aa", "part_ab"]:
    hf_hub_download(
        repo_id=index_repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=save_path,
    )

hf_hub_download(
    repo_id=args.corpus_repo_id,
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=save_path,
)

print(f"Downloaded retrieval files to {save_path}")
