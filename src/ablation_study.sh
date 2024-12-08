#!/bin/bash

### Code to reproduce ablation study. Please refer to ablation study of the main paper for more details. 

open_ai_api_key='<Insert your OpenAI API Key Here>'

base_path="<Set the path to the project directory here>"

run_all_unfiltered() {
    echo "Running unfiltered with num_ref=all"
    output_file="${base_path}data/generated_research_ideas/gen_hyp_ablation_all_ref_unfiltered.csv"
    python "${base_path}src/generation/generate_hypotheses.py" --all_ref True \
    --references "${base_path}data/dataset/raw_references.csv" \
    --target_papers "${base_path}data/dataset/ablation_target_papers.csv" \
    --output "$output_file" \
    --api_key "$open_ai_api_key"

    eval_file="${base_path}data/evaluated_research_ideas/eval_hyp_ablation_all_ref_unfiltered.csv"
    python "${base_path}src/evaluation/evaluate_hypotheses.py" --input "$output_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria novelty --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria feasibility --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
}

run_all_filtered() {
    echo "Running filtered with num_ref=all"
    output_file="${base_path}data/generated_research_ideas/gen_hyp_ablation_all_ref_filtered.csv"
    python "${base_path}src/generation/generate_hypotheses.py" --all_ref True \
    --references "${base_path}data/dataset/ablation_filtered_references.csv" \
    --target_papers "${base_path}data/dataset/ablation_target_papers.csv" \
    --output "$output_file" \
    --api_key "$open_ai_api_key"

    eval_file="${base_path}data/evaluated_research_ideas/eval_hyp_ablation_all_ref_filtered.csv"
    python "${base_path}src/evaluation/evaluate_hypotheses.py" --input "$output_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria novelty --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria feasibility --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
}

num_refs=(1 3 5 7 10 13 15)

# Function to run filtered evaluations
run_filtered() {
    for num_ref in "${num_refs[@]}"
    do  
        echo "Running filtered with num_ref=$num_ref"
        output_file="${base_path}data/generated_research_ideas/gen_hyp_ablation_num_ref_${num_ref}_filtered.csv"
        python "${base_path}src/generation/generate_hypotheses.py" --num_ref "$num_ref" \
        --references "${base_path}data/dataset/ablation_filtered_references.csv" \
        --target_papers "${base_path}data/dataset/ablation_target_papers.csv" \
        --output "$output_file" \
        --api_key "$open_ai_api_key"

        eval_file="${base_path}data/evaluated_research_ideas/eval_hyp_ablation_num_ref_${num_ref}_filtered.csv"
        python "${base_path}src/evaluation/evaluate_hypotheses.py" --input "$output_file" --output "$eval_file" --openai_api "$open_ai_api_key"
        python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria novelty --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
        python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria feasibility --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    done
}

# Function to run unfiltered evaluations
run_unfiltered() {
    for num_ref in "${num_refs[@]}"
    do
        echo "Running unfiltered with num_ref=$num_ref"
        output_file="${base_path}data/generated_research_ideas/gen_hyp_ablation_num_ref_${num_ref}_unfiltered.csv"
        python "${base_path}src/generation/generate_hypotheses.py" --num_ref "$num_ref" \
        --references "${base_path}data/dataset/raw_references.csv" \
        --target_papers "${base_path}data/dataset/ablation_target_papers.csv" \
        --output "$output_file" \
        --api_key "$open_ai_api_key"

        eval_file="${base_path}data/evaluated_research_ideas/eval_hyp_ablation_num_ref_${num_ref}_unfiltered.csv"
        python "${base_path}src/evaluation/evaluate_hypotheses.py" --input "$output_file" --output "$eval_file" --openai_api "$open_ai_api_key"
        python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria novelty --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
        python "${base_path}src/evaluation/llm_ranking_eval.py" --ranking_criteria feasibility --input "$eval_file" --output "$eval_file" --openai_api "$open_ai_api_key"
    done
}

# Run filtered and unfiltered in parallel
run_all_unfiltered &
run_all_filtered &
run_filtered &
run_unfiltered &

wait

