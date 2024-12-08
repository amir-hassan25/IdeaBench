import pandas as pd
import argparse


def load_data(references_file, target_papers_file):
    """Load the references and target papers data."""
    references_df = pd.read_csv(references_file)
    target_papers_df = pd.read_csv(target_papers_file)
    return references_df, target_papers_df


def filter_references(df):
    """Apply the filtering logic to the references dataframe."""
    # Remove rows without abstract or publicationTypes
    df = df.dropna(subset=['abstract', 'publicationTypes'])

    # Filter out rows where 'citationCount' is less than 5
    df = df[df['citationCount'] >= 5]
    df.reset_index(drop=True, inplace=True)

    # Removing irrelevant publication types
    exclude_publication_types = [
        'Review', 'Dataset', 'Editorial',
        'LettersAndComments', 'News', 'Book'
    ]
    df = df[~df['publicationTypes'].apply(
        lambda x: any(pub_type in x for pub_type in exclude_publication_types) if x is not None else False
    )]

    # Removing references that are only from the result or methodology section
    df = df[df['intents'].apply(lambda x: x != '[]' and x != "['result']" and x != "['methodology']")]
    df.reset_index(drop=True, inplace=True)

    # Removing references from target papers with less than 3 references
    target_paper_counts = df['targetPaperId'].value_counts()
    filtered_target_papers = target_paper_counts[target_paper_counts >= 3].index
    df = df[df['targetPaperId'].isin(filtered_target_papers)]
    df.reset_index(drop=True, inplace=True)

    return df


def filter_target_papers(df_cleaned, target_papers_df):
    """Filter out target papers that no longer have sufficient references."""
    target_papers = df_cleaned['targetPaperId'].unique()
    filtered_target_papers_df = target_papers_df[target_papers_df['paperId'].isin(target_papers)]
    filtered_target_papers_df.reset_index(drop=True, inplace=True)
    return filtered_target_papers_df


def ablation_study(df_cleaned, target_papers_df, max_references=15):
    # Filter out target papers with more than max_references
    target_paper_counts = df_cleaned['targetPaperId'].value_counts()
    filtered_target_papers = target_paper_counts[target_paper_counts <= max_references].index
    df_cleaned = df_cleaned[df_cleaned['targetPaperId'].isin(filtered_target_papers)]
    df_cleaned.reset_index(drop=True, inplace=True)

    # Filter the corresponding target papers
    filtered_target_papers_df = filter_target_papers(df_cleaned, target_papers_df)
    
    return df_cleaned, filtered_target_papers_df


def save_data(df_cleaned, filtered_target_papers_df, references_output, target_papers_output):
    """Save the filtered references and target papers data."""
    df_cleaned.to_csv(references_output, index=False)
    filtered_target_papers_df.to_csv(target_papers_output, index=False)
    print(f"Filtered references saved to {references_output}")
    print(f"Filtered target papers saved to {target_papers_output}")


def main(references_file, target_papers_file, references_output, target_papers_output, ablation=False, ablation_output=None):
    # Load data
    references_df, target_papers_df = load_data(references_file, target_papers_file)

    # Filter references
    df_cleaned = filter_references(references_df)

    # Filter target papers
    filtered_target_papers_df = filter_target_papers(df_cleaned, target_papers_df)

    # Save filtered data
    save_data(df_cleaned, filtered_target_papers_df, references_output, target_papers_output)

    if ablation:
        # create ablation data
        df_ablation, target_papers_ablation_df = ablation_study(df_cleaned, filtered_target_papers_df)
        
        # Save ablation data
        save_data(df_ablation, target_papers_ablation_df, ablation_output['references'], ablation_output['target_papers'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter references and adjust target papers.')
    parser.add_argument('--references', type=str, required=True, help='The path to the references.csv file.')
    parser.add_argument('--target_papers', type=str, required=True, help='The path to the target_papers.csv file.')
    parser.add_argument('--references_output', type=str, required=True, help='The path to save the filtered references.csv file.')
    parser.add_argument('--target_papers_output', type=str, required=True, help='The path to save the filtered target_papers.csv file.')
    parser.add_argument('--ablation', action='store_true', help='Include this flag to perform ablation study.')
    parser.add_argument('--ablation_references_output', type=str, help='The path to save the ablation filtered references.csv file.')
    parser.add_argument('--ablation_target_papers_output', type=str, help='The path to save the ablation filtered target_papers.csv file.')

    args = parser.parse_args()

    ablation_output = None
    if args.ablation:
        if not args.ablation_references_output or not args.ablation_target_papers_output:
            parser.error("Ablation outputs must be provided if ablation flag is set.")
        ablation_output = {
            'references': args.ablation_references_output,
            'target_papers': args.ablation_target_papers_output
        }

    main(args.references, args.target_papers, args.references_output, args.target_papers_output, args.ablation, ablation_output)