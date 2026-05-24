

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. We recommend using a separate environment.
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```


## Quick start

Train a reasoning + search LLM on NQ dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
# Run from the verl repository root.
save_path=examples/search_agent_rl/local_dense_retriever/search_data
python examples/search_agent_rl/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -dk $save_path/wiki-18.jsonl.gz
```

(2) Launch a local retrieval server.
```bash
conda activate retriever
bash examples/search_agent_rl/local_dense_retriever/start_retrieval.sh
```
