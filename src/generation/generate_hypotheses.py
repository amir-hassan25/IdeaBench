import pandas as pd
import os
import random
from tqdm import tqdm
import argparse
from transformers import pipeline  # Importing Hugging Face pipeline

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import DeepInfra
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms import ChatLlamaAPI
from langchain import FewShotPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from llamaapi import LlamaAPI
import openai
from langchain_groq import ChatGroq
from distutils.util import strtobool

import time

# CREDIT https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/01-langchain-prompt-templates.ipynb


parser = argparse.ArgumentParser(description="A description of what your script does.")
parser.add_argument('--abstract_sampling', type=str, default='standard')
parser.add_argument('--num_hyp', type=int, default=3)
parser.add_argument('--all_ref', type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument('--num_ref', type=int, default=1)
parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
parser.add_argument('--references', type=str)
parser.add_argument('--target_papers', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--api_key', type=str)
args = parser.parse_args()

if "llama" in args.model_name.lower() or "gemma" in args.model_name.lower() or "mixtral" in args.model_name.lower():
    if args.api_key is None:
        raise ValueError("Please provide an API key for the llama model")
    os.environ["DEEPINFRA_API_TOKEN"] = args.api_key
elif "gpt" in args.model_name.lower():
    if args.api_key is None:
        raise ValueError("Please provide an API key for the openai model")
    os.environ["OPENAI_API_KEY"] = args.api_key
elif "gemini" in args.model_name.lower():
    if args.api_key is None:
        raise ValueError("Please provide an API key for the google model")
    os.environ["GOOGLE_API_KEY"] = args.api_key


def prepare_model_pipeline(model_name):
    if "llama" in model_name.lower() or "gemma" in model_name.lower() or "mixtral" in model_name.lower():
        print('using llama model')

        if model_name == 'llama3.1-8b':
            llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")  
            llm.model_kwargs = {"repetition_penalty": 1.2}
            return llm
        elif model_name == 'llama3.1-70b':
            return DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")
        
        elif model_name == 'llama3.1-405b':
            print("using ", model_name)
            return DeepInfra(model_id="meta-llama/Meta-Llama-3.1-405B-Instruct")

    elif "gpt" in model_name.lower():
        return ChatOpenAI(model=model_name)
    elif "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(model=model_name)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def abstract_retrieval(target_paper_id, references_df, n):

    abstracts = references_df[references_df['targetPaperId'] == target_paper_id]['abstract'].tolist()
    # print(args.all_ref)  
    if args.all_ref == True:
        print('using all hypotheses')
        return abstracts
    if len(abstracts) < n:
        return abstracts  # If there are fewer than n abstracts, return all of them
    # print(n)
    sample_n_abstracts = random.sample(abstracts, n)
    assert len(sample_n_abstracts) == args.num_ref
    return sample_n_abstracts # Randomly sample n abstrac
    
# Util to prepare retrieved abstracts for the prompt
def get_abstracts_for_prompt(target_paper_id, references_df, n):

    abstracts = abstract_retrieval(target_paper_id, references_df, n)

    # Initialize an empty list to hold the dictionaries
    examples = []

    # Loop through the list of abstracts and create a dictionary for each
    for i, abstract in enumerate(abstracts):
        abstract_num = i + 1
        # Handle special characters in the abstract
        abstract = abstract.replace("{greater than or equal to}", "≥")
        abstract = abstract.replace("{", "")
        abstract = abstract.replace("}", "")
        example_dict = {"abstract_num": abstract_num,
                        "abstract": abstract}
        examples.append(example_dict)
    
    return examples

def create_prompt(examples):
    # create an example template
    example_template = """Abstract {abstract_num}: {abstract}"""

    # create a prompt example from the above template
    example_prompt = PromptTemplate(
        input_variables=["abstract", "abstract_num"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """You are a biomedical researcher. You are tasked with creating a hypothesis or research idea given some background knowledge. The background knowledge I will provide abstracts from other papers.

    Here are the abstracts:"""
    # and the suffix our user input and output indicator
    suffix = """Using these abstracts, reason over them and come up with a novel hypothesis. Please avoid copying ideas directly, rather use the insights to inspire a novel hypothesis in the form of a brief and concise paragraph."""

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        # input_variables=["query"],
        example_separator="\n\n"
    )
    return few_shot_prompt_template

# Hypothesis generation pipeline
def main(num_hypotheses, num_ref, model_name):
    # model = ChatOpenAI(model=model_name)
    model = prepare_model_pipeline(model_name=model_name)
    parser = StrOutputParser()

    # import target and reference papers
    # if args.abstract_sampling == 'random': 
    #     # references_df = pd.read_csv(args.data_dir+'/references.csv')
    #     # references_df= references_df.dropna(subset=['abstract'])
    #     assert 'random' in args.references
    # else: 
    references_df = pd.read_csv(args.references)
    references_df= references_df.dropna(subset=['abstract'])

    target_paper_df = pd.read_csv(args.target_papers).head(5) #REMOVE HEAD

    # Initialize a dictionary to store hypotheses for each target paper
    hypotheses_dict = {}

    for index, row in tqdm(target_paper_df.iterrows(), total=target_paper_df.shape[0], desc="Processing papers"):

        target_paper_id = row['paperId']
        # Logic to get paper id and create examples
        background_abstracts = get_abstracts_for_prompt(target_paper_id, references_df, num_ref)
        
        # Debug print statement to check the content of background_abstracts
        # print(f"Background abstracts for paper ID {target_paper_id}: {background_abstracts}")
        
        # Validate that background_abstracts contains the correct structure
        if not all('abstract_num' in ex and 'abstract' in ex for ex in background_abstracts):
            raise ValueError(f"Invalid structure in background abstracts for paper ID {target_paper_id}: {background_abstracts}")

        prompt = create_prompt(background_abstracts)
        # print(prompt.format())
        hypotheses = []
        # create chain
        chain = prompt | model | parser
        # for _ in range(num_hypotheses):
        #     try:
        #         # Try to invoke the chain and get a hypothesis
        #         hypotheses.append(chain.invoke({}))
        #     except Exception as e:
        #         time.sleep(30)
        #         try:
        #             hypotheses.append(chain.invoke({}))
        #         except Exception as e:
        #             # If it fails again, append the error message
        #             error_message = f"Error generating hypothesis for paper ID {target_paper_id}: {str(e)}"
        #             print(error_message)
        #             hypotheses.append(error_message)
        for _ in range(num_hypotheses):
            try:
                # Try to invoke the chain and get a hypothesis
                hypotheses.append(chain.invoke({}))
                
            except Exception as e:
                # Check if the error is a rate limit error
                error_message = str(e)
                if "rate_limit_exceeded" in error_message and "Requested" in error_message and "Limit" in error_message:
                    # Extract the requested and limit tokens
                    requested_tokens = int(error_message.split("Requested ")[1].split(".")[0])
                    limit_tokens = int(error_message.split("Limit ")[1].split(",")[0])
                    
                    # If the requested tokens exceed the limit, exit the loop
                    if requested_tokens > limit_tokens:
                        print("Requested tokens exceed the limit. Exiting loop.")
                        error_message = f"Error generating hypothesis for paper ID {target_paper_id}: {str(e)}"
                        hypotheses.append(error_message)
                        continue

                # If the error is not due to the rate limit, try again after sleeping
                time.sleep(75)
                try:
                    hypotheses.append(chain.invoke({}))
                except Exception as e:
                    # If it fails again, append the error message
                    error_message = f"Error generating hypothesis for paper ID {target_paper_id}: {str(e)}"
                    print(error_message)
                    hypotheses.append(error_message)

        # print(hypotheses)
        # Store hypotheses in the dictionary
        hypotheses_dict[target_paper_id] = hypotheses
        
    # Add the hypotheses as a new column in target_paper_df
    target_paper_df['hypotheses'] = target_paper_df['paperId'].map(hypotheses_dict)
    
    # Save the updated DataFrame to a new CSV file
    target_paper_df.to_csv(args.output, index=False)
    print(f"Hypotheses added and saved to {args.output}")

# Run the main function
if __name__ == "__main__":
    print("Num hypotheses to generate: ", args.num_hyp)
    # print("Num references to sample: ", args.num_ref)
    print("Model name to generate hypotheses: ", args.model_name)
    print(args)
    main(num_hypotheses=args.num_hyp, num_ref=args.num_ref, model_name=args.model_name)


