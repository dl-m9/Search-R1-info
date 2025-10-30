export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com

file_path=/forest/forest/Search-R1-info/corpus
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2


nohup python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu > retrieval.log 2>&1 &
