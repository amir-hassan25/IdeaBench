import os
import time
import pandas as pd
import evaluate
from bert_score import BERTScorer
from ast import literal_eval
from tqdm import tqdm
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

parser = argparse.ArgumentParser(description="A description of what your script does.")
parser.add_argument('--llm_rating_eval', type=bool, default=True)
parser.add_argument('--llm_ranking_eval', type=bool, default=False)
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--openai_api', type=str)
args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api


class HypothesisEvaluator:
    def __init__(self, do_llm_eval=False):
        # Load BERTScorer
        self.scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
        print(self.scorer.device)
        # Load the ROUGE and BLEU evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        # Initialize the LLM evaluator
        self.llm_evaluator = LLMEvaluator()
        self.do_llm_eval = do_llm_eval

    # Evaluate for only the hypothesis with the highest bertf1 score
    def evaluate(self, hypotheses, target_hypothesis):
        P, R, F1 = self.scorer.score(hypotheses, [target_hypothesis] * len(hypotheses))
        best_f1_idx = F1.argmax()
        best_hypothesis = hypotheses[best_f1_idx]

        rouge_result = self.rouge.compute(predictions=[best_hypothesis], references=[target_hypothesis])
        bleu_result = self.bleu.compute(predictions=[best_hypothesis], references=[target_hypothesis])

        if self.do_llm_eval:
            llm_evaluation = self.llm_evaluator.evaluate_with_retry(best_hypothesis, target_hypothesis)
        else:
            llm_evaluation = None

        return {
            'hypothesis': best_hypothesis,
            'bert_score': {
                'precision': P[best_f1_idx].item(),
                'recall': R[best_f1_idx].item(),
                'f1': F1[best_f1_idx].item()
            },
            'rouge_score': rouge_result,
            'bleu_score': bleu_result,
            'llm_evaluation': llm_evaluation
        }


class LLMEvaluator:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")
        self.parser = StrOutputParser()
        self.prompt_template = PromptTemplate(
            input_variables=["hypothesis", "abstract"],
            template="""You are an expert in understanding and analyzing scientific content. Your task is to evaluate the degree of overlap between the ideas presented in a hypothesis and the abstract of a scientific paper. Please read both the hypothesis and the abstract carefully. Then, rate the overlap on a scale of 1 to 10, where 1 indicates minimal or no overlap, and 10 indicates a perfect or nearly perfect overlap. Provide a brief explanation for your rating.

Hypothesis: {hypothesis}

Abstract: {abstract}

Rating:
On a scale of 1-10, rate the overlap between the ideas in the hypothesis and the abstract.

Explanation:
In one sentence, provide a brief explanation for your rating, mentioning the key points of overlap and any significant differences you observed."""
        )

    def evaluate(self, hypothesis, target_hypothesis):
        chain = self.prompt_template | self.model | self.parser
        llm_evaluation = chain.invoke({'hypothesis': hypothesis, 'abstract': target_hypothesis})
        try:
            rating_part = llm_evaluation.split('Rating:')[1].strip()
            explanation_part = rating_part.split('Explanation:')[1].strip()
            llm_rating = rating_part.split('Explanation:')[0].strip()
            return {'rating': llm_rating, 'explanation': explanation_part}
        except IndexError:
            return {'rating': 'N/A', 'explanation': llm_evaluation}

    def evaluate_with_retry(self, hypothesis, target_hypothesis, max_retries=3, delay=10):
        for attempt in range(max_retries):
            try:
                return self.evaluate(hypothesis, target_hypothesis)
            except Exception as e:
                if "Rate limit exceeded" in str(e):
                    if attempt < max_retries - 1:
                        print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries reached. Returning default response.")
                        return {'rating': 'N/A', 'explanation': 'Rate limit exceeded after multiple attempts.'}
                else:
                    raise e


def generate_metrics_for_dataframe(df, do_llm_eval):
    evaluator = HypothesisEvaluator(do_llm_eval=do_llm_eval)
    metrics_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            hypotheses = literal_eval(row['hypotheses'])
            abstract = row['abstract']
            metrics = evaluator.evaluate(hypotheses, abstract)
            metrics_list.append(metrics)
        except Exception as e:
            print(str(e))  # logging error
            metrics = {'error': str(e)}
            metrics_list.append(metrics)

    df_copy = df.copy()
    df_copy['metrics'] = metrics_list
    return df_copy


def main(do_llm_eval):
    # df with generated hypotheses
    df = pd.read_csv(args.input)
    print("loaded df with generated hypotheses")

    df_with_metrics = generate_metrics_for_dataframe(df, do_llm_eval)
    print("Completed evaluation")

    df_with_metrics.to_csv(args.output)
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main(args.llm_rating_eval)