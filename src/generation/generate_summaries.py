import pandas as pd
import os
import random
from tqdm import tqdm
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import FewShotPromptTemplate


# CREDIT https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/01-langchain-prompt-templates.ipynb

parser = argparse.ArgumentParser(description="Generates abstract summaries for target papers")
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--api_key', type=str)
args = parser.parse_args()
os.environ["OPENAI_API_KEY"] = args.api_key

class SummaryGenerator:
    def __init__(self):
        
        self.model = ChatOpenAI(model="gpt-4o")
        self.parser = StrOutputParser()
        self.prompt_template = PromptTemplate(
            input_variables=["abstract"],
            template="""Write a concise paragraph summarizing the following biomedical paper abstract as if you are proposing your own research idea or hypothesis. Focus on describing the main research idea and provide a high-level summary of the findings without detailed results or specific numerical data. Please begin the paragraph with "Hypothesis: " or "Given that ".  

Abstract:
{abstract}

Summary: """
        )
        self.chain = self.prompt_template | self.model | self.parser


    def gen(self, abstract):
        summary = self.chain.invoke({'abstract': abstract})
        return summary
    
# summery generation pipeline
def main(model_name="gpt-4o"):

    target_paper_df = pd.read_csv(args.input)
    summary_generator = SummaryGenerator()
    summary_dict = {}

    for index, row in tqdm(target_paper_df.iterrows(), total=target_paper_df.shape[0], desc="Processing papers"):

        target_paper_abstract = row['abstract']
        target_paper_id = row['paperId']

        try:
            summary = summary_generator.gen(target_paper_abstract)
            summary_dict[target_paper_id] = summary
        except Exception as e:
            print(str(e))
            summary_dict[target_paper_id] = f'err occured: {str(e)}'

        
    # Add the hypotheses as a new column in target_paper_df
    target_paper_df['abstract_summary'] = target_paper_df['paperId'].map(summary_dict)
    
    # Save the updated DataFrame to a new CSV file
    target_paper_df.to_csv(args.output, index=False)
    print(f"Hypotheses added and saved to {args.output}")


# Run the main function
if __name__ == "__main__":
    main()