import requests
import pandas as pd
import time
import argparse


def get_references(paper_id, offset=0, limit=300, fields=None):
    """Fetch references of a given paper from the Semantic Scholar API."""
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params = {'offset': offset, 'limit': limit}

    if fields:
        params['fields'] = fields

    response = requests.get(base_url, params=params)
    
    # Handle rate limiting
    if response.status_code == 429:
        print("Rate limit exceeded. Waiting for 1 second before retrying...")
        time.sleep(1)
        return get_references(paper_id, offset, limit, fields)  # Retry after delay

    response.raise_for_status()  # Raise an exception for other HTTP errors
    return response.json()


def retrieve_references(target_papers, fields):
    """Retrieve references for a list of target papers."""
    reference_df_list = []
    
    for target_paper_id in target_papers['paperId']:
        data = get_references(target_paper_id, fields=fields)
        
        # Check if 'data' contains references
        if 'data' in data:
            references = data['data']
            for ref in references:
                cited_paper = ref.get('citedPaper', {})
                ref_data = {
                    'targetPaperId': target_paper_id,
                    'contexts': ref.get('contexts', []),
                    'intents': ref.get('intents', []),
                    'isInfluential': ref.get('isInfluential', False),
                    'paperId': cited_paper.get('paperId', None),
                    'externalIds': cited_paper.get('externalIds', {}),
                    'title': cited_paper.get('title', None),
                    'abstract': cited_paper.get('abstract', None),
                    'venue': cited_paper.get('venue', None),
                    'year': cited_paper.get('year', None),
                    'citationCount': cited_paper.get('citationCount', 0),
                    'publicationTypes': cited_paper.get('publicationTypes', [])
                }
                reference_df_list.append(ref_data)
        else:
            print(f"No references found for paper ID: {target_paper_id}")
        
        # Wait for 1 second before the next request to avoid rate limiting
        time.sleep(1)

    # Convert list of dictionaries to a DataFrame
    reference_df = pd.DataFrame(reference_df_list)
    return reference_df


def main(input_file, output_file):
    # Load target papers
    target_papers = pd.read_csv(input_file)

    # Define fields to retrieve for each reference
    fields = "paperId,title,abstract,year,venue,contexts,intents,citationCount,publicationTypes,externalIds,isInfluential"

    # Retrieve references
    reference_df = retrieve_references(target_papers, fields)

    # Save the references to a CSV file
    reference_df.to_csv(output_file, index=False)
    print(f"References saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve references of target papers and save to a CSV file.')
    parser.add_argument('--input', type=str, required=True, help='The path to the target_papers.csv file.')
    parser.add_argument('--output', type=str, required=True, help='The path to save the references.csv file.')

    args = parser.parse_args()

    main(args.input, args.output)