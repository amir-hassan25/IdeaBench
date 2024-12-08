
import pandas as pd
from ast import literal_eval
import re
import os
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model performance and save results to a file.")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model to evaluate.")
    parser.add_argument('--high_ref_file', type=str, required=True, help="Path to the high reference CSV file.")
    parser.add_argument('--low_ref_file', type=str, required=True, help="Path to the low reference CSV file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the evaluation results.")
    return parser.parse_args()

def calculate_mean_median(bert_scores, llm_ratings):
    mean_bert_scores = round(bert_scores['f1'].mean(), 3)
    median_bert_scores = round(bert_scores['f1'].median(), 3)
    percentile_80_bert_scores = round(bert_scores['f1'].quantile(0.8), 3)
    
    mean_llm_rating = round(llm_ratings.mean(), 3)
    median_llm_rating = round(llm_ratings.median(), 3)
    percentile_80_llm_rating = round(llm_ratings.quantile(0.8), 3)
    
    return mean_bert_scores, median_bert_scores, percentile_80_bert_scores, mean_llm_rating, median_llm_rating, percentile_80_llm_rating

def extract_llm_rating(rating_str):
    match = re.search(r'\d+', rating_str)
    if match:
        return int(match.group())
    else:
        return None

def extract_metrics(df):
    bert_scores = {'precision': [], 'recall': [], 'f1': []}
    llm_ratings = []
    
    for idx in range(len(df)):
        try:
            metrics = literal_eval(df['metrics'].iloc[idx])
            bert_scores['precision'].append(metrics['bert_score']['precision'])
            bert_scores['recall'].append(metrics['bert_score']['recall'])
            bert_scores['f1'].append(metrics['bert_score']['f1'])
            llm_rating = extract_llm_rating(metrics['llm_evaluation']['rating'])
            if llm_rating is not None:
                llm_ratings.append(llm_rating)
            else:
                llm_rating = extract_llm_rating(metrics['llm_evaluation']['explanation'])
                llm_ratings.append(llm_rating)
        except KeyError:
            continue
        
    return pd.DataFrame(bert_scores), pd.Series(llm_ratings)

def print_scores_for_model(dataframe, model_name, output_file):
    bert_scores, llm_ratings = extract_metrics(dataframe)
    
    mean_median_percentiles = calculate_mean_median(bert_scores, llm_ratings)
    
    with open(output_file, 'a') as f:
        f.write(f"\n{model_name} Mean, Median, and 80th Percentile:\n")
        f.write(f"Mean BERTScore: {mean_median_percentiles[0]}\n")
        f.write(f"Median BERTScore: {mean_median_percentiles[1]}\n")
        f.write(f"80th Percentile BERTScore: {mean_median_percentiles[2]}\n")
        f.write(f"Mean LLM Rating: {mean_median_percentiles[3]}\n")
        f.write(f"Median LLM Rating: {mean_median_percentiles[4]}\n")
        f.write(f"80th Percentile LLM Rating: {mean_median_percentiles[5]}\n")

def extract_hypothesis_order(text):
    pattern = r'\d+\.\s\*\*Hypothesis ([A-Z])\*\*'
    matches = re.findall(pattern, text)
    return list(matches)

def update_ranking_list(row):
    lst, text = literal_eval(row)
    if not lst:
        return extract_hypothesis_order(text)
    return lst

def get_relative_rank(ranking):
    if "A" in ranking:
        numerator = ranking.index('A')
        denominator = len(ranking) - 1
        if denominator == 0:
            return 0
        rel_rank = numerator / denominator
        return rel_rank
    else:
        return None

def avg_rel_rank(rankings):
    relative_ranks = [get_relative_rank(row) for row in rankings]
    relative_ranks_array = np.array(relative_ranks)
    cleaned_array = relative_ranks_array[relative_ranks_array != None]

    mean = np.mean(cleaned_array)
    median = np.median(cleaned_array)
    percentile_75th = np.percentile(cleaned_array, 75)

    return mean

def main():
    args = parse_arguments()
    
    # Read the CSV files
    df_high = pd.read_csv(args.high_ref_file)
    df_low = pd.read_csv(args.low_ref_file)
    
    # Update ranking lists
    df_high['llm_novelty_ranking_eval'] = df_high['llm_novelty_ranking_eval'].apply(update_ranking_list)
    df_high['llm_feasibility_ranking_eval'] = df_high['llm_feasibility_ranking_eval'].apply(update_ranking_list)
    df_low['llm_novelty_ranking_eval'] = df_low['llm_novelty_ranking_eval'].apply(update_ranking_list)
    df_low['llm_feasibility_ranking_eval'] = df_low['llm_feasibility_ranking_eval'].apply(update_ranking_list)
    
    # Evaluate and save results
    print_scores_for_model(df_high, f"{args.model_name} High", args.output_file)
    print_scores_for_model(df_low, f"{args.model_name} Low", args.output_file)
    
    # Compute and print insight scores
    with open(args.output_file, 'a') as f:
        f.write(f'\nModel: {args.model_name}\n')
        
        novelty_avg_rel_rank_high = avg_rel_rank(df_high['llm_novelty_ranking_eval'])
        feasibility_avg_rel_rank_high = avg_rel_rank(df_high['llm_feasibility_ranking_eval'])
        f.write(f"High Reference - Insight Score (novelty): {novelty_avg_rel_rank_high}\n")
        f.write(f"High Reference - Insight Score (feasibility): {feasibility_avg_rel_rank_high}\n")
        
        novelty_avg_rel_rank_low = avg_rel_rank(df_low['llm_novelty_ranking_eval'])
        feasibility_avg_rel_rank_low = avg_rel_rank(df_low['llm_feasibility_ranking_eval'])
        f.write(f"Low Reference - Insight Score (novelty): {novelty_avg_rel_rank_low}\n")
        f.write(f"Low Reference - Insight Score (feasibility): {feasibility_avg_rel_rank_low}\n")

if __name__ == "__main__":
    main()


