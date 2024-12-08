#!/bin/bash


### Code to analyze evaluation of research ideas in order to get the similarity and LLM-based metrics of the research ideas.

# Base path of project directory 
base_path="<Set the path to the project directory here>"


# Define the output file
output_file="all_model_evaluations.txt"

# Clear the output file if it already exists
> "$output_file"

# Function to evaluate a model and append results to the output file
evaluate_model() {
    model_name=$1
    high_ref_file=$2
    low_ref_file=$3

    # Run the Python script to evaluate the model and append to the output file
    python "${base_path}src/evaluation/analyze_results.py" --model_name "$model_name" --high_ref_file "$high_ref_file" --low_ref_file "$low_ref_file" --output_file "$output_file"
}

# List of models and their corresponding file paths
# Add more models as needed
models=(
    "<MODEL_NAME>:<PATH_TO_HIGH_RESOURCE_EVAL.csv>:<PATH_TO_LOW_RESOURCE_EVAL.csv>"
)

# Example of how to set up
# models=(
#     "Llamma3.1-70b:/../IdeaBench/data/evaluated_research_ideas/eval_hyp_llama3.1-70b_all_ref_True_filtered_ref_True_hyp_3_ref_3.csv:/../IdeaBench/data/evaluated_research_ideas/eval_hyp_llama3.1-70b_all_ref_False_filtered_ref_True_hyp_3_ref_3.csv"
#     "gemini-1.5-pro:/../IdeaBench/data/evaluated_research_ideas/eval_hyp_gemini-1.5-pro_all_ref_True_filtered_ref_True_hyp_3_ref_3.csv:/../IdeaBench/data/evaluated_research_ideas/eval_hyp_gemini-1.5-pro_all_ref_False_filtered_ref_True_hyp_3_ref_3.csv
# )

# Loop over the models and evaluate each one
for model in "${models[@]}"; do
    IFS=":" read -r model_name high_ref_file low_ref_file <<< "$model"
    echo "Evaluating $model_name..."
    evaluate_model "$model_name" "$high_ref_file" "$low_ref_file"
done

echo "All model evaluations saved to $output_file"