import requests
import pandas as pd
import argparse


def fetch_papers(url, params, token=None):
    """Fetch papers from the Semantic Scholar API."""
    if token:
        params['token'] = token
    response = requests.get(url, params=params)
    response.raise_for_status()  # Check for request errors
    return response.json()


def retrieve_papers(url, params, strategy, debug=False):
    """Retrieve and process papers based on the provided parameters."""
    columns = ['paperId', 'title', 'abstract', 'year', 'venue', 'citationCount', 'publicationTypes', 'externalIds']
    papers_df = pd.DataFrame(columns=columns)
    total_papers = 0
    max_papers = 50 if debug else float('inf')  # Limit to 50 papers if in debug mode

    data = fetch_papers(url, params)
    papers_df = pd.concat([papers_df, pd.DataFrame(data['data'])], ignore_index=True)
    total_papers += len(data['data'])

    # Fetch subsequent batches if token is present and haven't reached the debug limit
    while 'token' in data and data['token'] and total_papers < max_papers:
        token = data['token']
        data = fetch_papers(url, params, token)
        papers_df = pd.concat([papers_df, pd.DataFrame(data['data'])], ignore_index=True)
        total_papers += len(data['data'])

    # Limit the dataframe to max_papers rows if in debug mode
    if debug:
        papers_df = papers_df.head(max_papers)

    # Remove rows where the abstract is None
    papers_df.dropna(subset=['abstract'], inplace=True)
    papers_df['strategy'] = strategy

    return papers_df


def main(year, output_file, debug=False):
    # Define the common URL for API requests
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    # Strategy 1: Retrieve papers from top venues
    # Venues chosen from Google Scholar 
    top_venues = 'Nature,Science,Nature Communications,Cell,Proceedings of the National Academy of Sciences,International Journal of Molecular Sciences,Nucleic Acids Research,PLOS ONE,Science Advances,Scientific Reports,Nature Biotechnology,Nature Genetics,Nature Methods,New England Journal of Medicine,The Lancet,Cell,JAMA,Nature Medicine,Proceedings of the National Academy of Sciences,International Journal of Molecular Sciences,PLOS ONE,BMJ,Frontiers in Immunology,Journal of Clinical Oncology,Circulation,Nutrients,The Lancet Oncology,Morbidity and Mortality Weekly Report,Journal of the American College of Cardiology,Nature Genetics,Frontiers in Psychology'
    
    params_top_venues = {
        "fields": "paperId,title,abstract,year,venue,citationCount,publicationTypes,externalIds",
        "minCitationCount": "1",
        "year": year,
        "venue": top_venues,
        "sort": "citationCount:desc"
    }
    
    papers_df1 = retrieve_papers(url, params_top_venues, strategy=1, debug=debug)

    # Strategy 2: Retrieve papers from any venue but have at least 20 citations
    params_all_venues = {
        "fields": "paperId,title,abstract,year,venue,citationCount,publicationTypes,externalIds",
        "minCitationCount": "20",
        "year": year,
        "fieldsOfStudy": "Medicine,Biology",
        "sort": "citationCount:desc"
    }
    
    papers_df2 = retrieve_papers(url, params_all_venues, strategy=2, debug=debug)

    # Combine both strategies
    combined_df = pd.concat([papers_df1, papers_df2])
    combined_df.drop_duplicates(subset=['paperId'], keep='first', inplace=True)

    # Remove review articles and other unwanted publication types
    exclude_publication_types = ['Review', 'Dataset', 'Editorial', 'LettersAndComments', 'News', 'Book']
    combined_df = combined_df[~combined_df['publicationTypes'].apply(
        lambda x: any(pub_type in x for pub_type in exclude_publication_types) if x is not None else False
    )]

    # Save to CSV
    print(output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Target papers saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve target papers based on specified year and save to a CSV file.')
    parser.add_argument('--year', type=int, required=True, help='The publication year of the papers.')
    parser.add_argument('--output', type=str, required=True, help='The path to save the target_papers.csv file.')
    parser.add_argument('--debug', action='store_true', help='If set, limit retrieval to 50 papers.')

    args = parser.parse_args()

    main(args.year, args.output, debug=args.debug)