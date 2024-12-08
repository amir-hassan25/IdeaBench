import pandas as pd
import os
import re
import random
from tqdm import tqdm
from ast import literal_eval
import argparse
import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import FewShotPromptTemplate

# CREDIT https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/01-langchain-prompt-templates.ipynb

parser = argparse.ArgumentParser(description="A description of what >your script does.")
parser.add_argument('--ranking_criteria', type=str, default=f'novelty')
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--openai_api', type=str)
args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api


def remove_empty_strings(input_list):
    return [item for item in input_list if item != ""]

# int to letter utility for ranking prompt
def int_to_letter(n):
    return chr(ord('A') + n - 1)


# Util for preparing hypotheses for LLM Ranking Prompt for any number of hypotheses
def prepare_candidates_for_prompt(ground_truth, hypotheses):
    # Initialize an empty list to hold the dictionaries
    examples = []
    
    # Add ground truth to be the first. 
    examples.append(
        {
            "hypothesis": f"Hypothesis A: \n {ground_truth}"
        }
    )
    
    # Loop through the list of abstracts and create a dictionary for each
    for i, abstract in enumerate(hypotheses):
        hyp_num = i + 2
        letter = int_to_letter(hyp_num)
        example_dict = {"hypothesis": f"Hypothesis {letter}: \n {hypotheses[i]}"}
        examples.append(example_dict)
    
    return examples


def create_prompt(examples):
    # create an example template
    example_template = """{hypothesis}"""

    # create a prompt example from the above template
    example_prompt = PromptTemplate(
        input_variables=["hypothesis"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """You are a reviewer tasked with ranking the quality of a set of research ideas based on their {ranking_criteria}. The idea with the highest {ranking_criteria} should be ranked first. 

Please rank the following hypotheses in the format:

1. Hypothesis (insert letter): (insert brief rationale)
....
n. Hypothesis (insert letter): (insert brief rationale)

Please rank the following hypotheses: \n
"""
    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="",
        input_variables=["ranking_criteria"],
        example_separator="\n\n"
    )
    return few_shot_prompt_template

class LLMRankingEval:
    def __init__(self):
        
        self.model = ChatOpenAI(model="gpt-4o")
        self.parser = StrOutputParser()

    def parse_rankings(self, gen_text): 
        # Regular expression to find the hypothesis order
        pattern = r'Hypothesis ([A-Z]):'
        # Find all matches
        matches = re.findall(pattern, gen_text)
        # Convert matches to a list
        hypothesis_order = list(matches)
        return hypothesis_order   
    
    def gen(self, ranking_criteria, ground_truth, hypotheses):
        hypotheses = remove_empty_strings(hypotheses)
        candidates = prepare_candidates_for_prompt(ground_truth, hypotheses)
        prompt = create_prompt(candidates)
        formatted_prompt = prompt.format(ranking_criteria=ranking_criteria)
        # print(formatted_prompt)
    
        
        chain = prompt | self.model | self.parser
        # langchain.verbose = True

        gen_ranking_eval = chain.invoke({"ranking_criteria":ranking_criteria})
        ranked_eval = self.parse_rankings(gen_ranking_eval)
        return ranked_eval, gen_ranking_eval
    
 
# summery generation pipeline
def main(model_name="gpt-4o"):

    hypotheses_w_summaries_df = pd.read_csv(args.input)
    llm_eval_ranker = LLMRankingEval()

    ranking_eval_dict = {}

    for index, row in tqdm(hypotheses_w_summaries_df.iterrows(), total=hypotheses_w_summaries_df.shape[0], desc="Processing papers"):

        ground_truth = row['abstract_summary']
        target_paper_id = row['paperId']
        ranking_criteria = args.ranking_criteria
        hypotheses = literal_eval(hypotheses_w_summaries_df['hypotheses'].iloc[index])
        try:
            summary = llm_eval_ranker.gen(ranking_criteria=ranking_criteria, ground_truth=ground_truth, hypotheses=hypotheses)
            # print(summary)
            ranking_eval_dict[target_paper_id] = summary
        except Exception as e:
            # raise e
            print(str(e))
            ranking_eval_dict[target_paper_id] = ([],f'err occured: {str(e)}')


    # Add the hypotheses as a new column in target_paper_df
    hypotheses_w_summaries_df[f'llm_{args.ranking_criteria}_ranking_eval'] = hypotheses_w_summaries_df['paperId'].map(ranking_eval_dict)
    
    # Save the updated DataFrame to a new CSV file
    hypotheses_w_summaries_df.to_csv(args.output, index=False)
    print(f"LLM eval rankings added and saved to {args.output}")


# Run the main function
if __name__ == "__main__":
    main()