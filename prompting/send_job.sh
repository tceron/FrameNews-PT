#!/bin/bash

# Define model and prompt combinations
declare -a MODELS=(
    "gpt-4o-2024-08-06"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    # "google/gemma-3-27b-it"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

declare -a PROMPTS=("zero1" "zero2" "zero3_noex") 

# Loop over models
for model in "${MODELS[@]}"; do
  # Loop over prompts
  for prompt in "${PROMPTS[@]}"; do
    echo "Running model: $model with prompt: $prompt"
    python3 classification_with_model_prompting.py -m "$model" -p "$prompt"
  done
done