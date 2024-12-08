#!/bin/bash


### Code to build the benchmark dataset. Please refere to the Dataset Construction section of the main paper for more details. 

# OpenAI API Key for generating summaries or target paper abstracts
open_ai_api_key='<Insert your OpenAI API Key Here>'

# Year to extract target papers from
year=2024

base_path="<Set the path to the project directory here>."

# Set up paths by appending filenames to the base path
target_papers_path="${base_path}data/dataset/target_papers.csv"
ablation_target_papers_path="${base_path}data/dataset/ablation_target_papers.csv"
raw_reference_papers_path="${base_path}data/dataset/raw_references.csv"
filtered_reference_papers_path="${base_path}data/dataset/filtered_references.csv"
ablation_filtered_reference_papers_path="${base_path}data/dataset/ablation_filtered_references.csv"


echo "$target_papers_path"
python "${base_path}src/dataset/get_target_papers.py" --year "$year" --output "$target_papers_path" 
python "${base_path}src/dataset/retrieve_references.py" --input $target_papers_path --output $raw_reference_papers_path # This may take a while 
python "${base_path}src/dataset/filter_references.py" --references $raw_reference_papers_path \
    --target_papers $target_papers_path \
    --references_output $filtered_reference_papers_path \
    --target_papers_output $target_papers_path --ablation \
    --ablation_references_output $ablation_filtered_reference_papers_path \
    --ablation_target_papers_output $ablation_target_papers_path 
python "${base_path}src/generation/generate_summaries.py" --input $target_papers_path --output $target_papers_path --api_key $open_ai_api_key # This may take a while 
python "${base_path}src/generation/generate_summaries.py" --input $ablation_target_papers_path --output $ablation_target_papers_path --api_key $open_ai_api_key # This may take a while 

