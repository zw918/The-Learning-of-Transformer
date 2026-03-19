#!/bin/bash

export MINERU_MODEL_SOURCE=modelscope
export CUDA_VISIBLE_DEVICES=2

input_dir="/zzw/zzw/FullLLM-main/The-Learning-of-Transformer/02_RAG_ing/MinerU/data"
output_dir="/zzw/zzw/FullLLM-main/The-Learning-of-Transformer/02_RAG_ing/MinerU/outputs"

for file in "$input_dir"/*; do
    echo "Processing: $file"
    mineru -p "$file" -o "$output_dir"
done