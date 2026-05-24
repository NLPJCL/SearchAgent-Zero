# file_path=/mnt/bn/ttsa-relevance-2/jcli/github/open_verl/examples/sglang_multiturn/search_data/wikipedia/wiki-18.jsonl
index_file=/mnt/bn/ttsa-relevance-2/jcli/github/open_verl/examples/sglang_multiturn/search_data/prebuilt_indices/e5_Flat.index
corpus_file=/mnt/bn/ttsa-relevance-2/jcli/github/open_verl/examples/sglang_multiturn/search_data/wikipedia/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python /mnt/bn/ttsa-relevance-2/jcli/github/new_verl/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py\
#         --index_path $index_file \
#         --corpus_path $corpus_file \
#         --topk 3 \
#         --retriever_name $retriever_name \
#         --retriever_model $retriever_path \
#         --faiss_gpu

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python /mnt/bn/ttsa-relevance-2/jcli/github/new_verl/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py\
#         --index_path $index_file \
#         --corpus_path $corpus_file \
#         --topk 3 \
#         --retriever_name $retriever_name \
#         --retriever_model $retriever_path \
#         --faiss_gpu

CUDA_VISIBLE_DEVICES="0,1,2,3" python /mnt/bn/ttsa-relevance-2/jcli/github/new_verl/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py\
        --index_path $index_file \
        --corpus_path $corpus_file \
        --topk 3 \
        --retriever_name $retriever_name \
        --retriever_model $retriever_path \
        --faiss_gpu