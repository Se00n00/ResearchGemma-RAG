# For List of All Evaluation metrics, Check: https://github.com/Se00n00/ResearchGemma-RAG/blob/main/docs/evaluation_metrics.md

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import NonLLMContextPrecisionWithReference, LLMContextPrecisionWithReference, NonLLMContextRecall, LLMContextRecall
from ragas.metrics.collections import RougeScore, BleuScore, ExactMatch, NonLLMStringSimilarity, DistanceMeasure

from langchain_openai import ChatOpenAI

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

LLM = os.getenv("LLM")
BASE_URL = os.getenv("BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatOpenAI(
    model = LLM,
    api_key = GROQ_API_KEY,
    base_url = BASE_URL,
    streaming = True
)

evaluator_llm = LangchainLLMWrapper(llm)

# --------------------------------------------------------------------------- #
#                               Context Recall                       LLM-Based       
# --------------------------------------------------------------------------- #
context_recall_judge = LLMContextRecall(llm=evaluator_llm)
async def llm_context_recall(inputs:dict, outputs:dict, reference_outputs:dict):
    """
    Measures how many of the relvant documents were successfully retreived.
    return : 0 - 1
    """
    example = SingleTurnSample(
        user_input = inputs['input'],
        response = outputs['answer'],
        reference = reference_outputs['expected_output'],
        retrieved_contexts = outputs['context'][0],
    )
    return await context_recall_judge.single_turn_ascore(example)

# --------------------------------------------------------------------------- #
#                               Context Recall                       Non-LLM-Based       
# --------------------------------------------------------------------------- #
context_recall = NonLLMContextRecall()
async def non_llm_context_recall(inputs:dict, outputs:dict, Metadata):
    example = SingleTurnSample(
        retrieved_contexts = outputs['context'][0],
        reference_contexts = Metadata['meta_data']['evidence']
    )
    return await context_recall.single_turn_ascore(example)


# --------------------------------------------------------------------------- #
#                               Context Precision                    LLM-Based       
# --------------------------------------------------------------------------- #

context_precision_judge = LLMContextPrecisionWithReference(llm=evaluator_llm)
async def llm_context_precision(inputs:dict, outputs:dict, reference_outputs:dict):
    """
    A llm based methods to determine wheather a retreived context is relevant

    values used for measuring:
        user_input: str
        reference: str
        reference_contexts: list[str]
    """
    example = SingleTurnSample(
        user_input = inputs['input'],
        reference = reference_outputs['expected_output'],
        retrieved_contexts = outputs['context'][0],
    )

    return await context_precision_judge.single_turn_ascore(example)


# --------------------------------------------------------------------------- #
#                               Context Precision                    Non-LLM-Based       
# --------------------------------------------------------------------------- #

context_precision = NonLLMContextPrecisionWithReference()
async def non_llm_context_precision(inputs:dict, outputs:dict, Metadata):
    """
    A non-llm based methods to determine wheather a retreived context is relevant

    values used for measuring:
        retirieved_contexts: list[str]
        reference_contexts: list[str]
    """
    example = SingleTurnSample(
        retrieved_contexts = outputs['context'][0],
        reference_contexts = Metadata['meta_data']['evidence']
    )
    return await context_precision.single_turn_ascore(example)

# --------------------------------------------------------------------------- #
#                               nDCG@K       
# --------------------------------------------------------------------------- #
 # TOBE Implemented in future

# --------------------------------------------------------------------------- #
#                               MRR@K       
# --------------------------------------------------------------------------- #
 # TOBE Implemented in future

# --------------------------------------------------------------------------- #
#                               Exact Match       
# --------------------------------------------------------------------------- #
em_scorer = ExactMatch()
async def EM(inputs:dict, outputs:dict, reference_outputs:dict):
    result = await em_scorer.ascore(
        reference = reference_outputs['expected_output'],
        response = outputs['answer'] 
    )

    return result.value

# --------------------------------------------------------------------------- #
#                               LLM String Similarity       
# --------------------------------------------------------------------------- #
SS_scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)
async def String_Similarity(inputs:dict, outputs:dict, reference_outputs:dict):
    result = await SS_scorer.ascore(
        reference = reference_outputs['expected_output'],
        response = outputs['answer'] 
    )

    return result.value

# --------------------------------------------------------------------------- #
#                               BLEU Score       
# --------------------------------------------------------------------------- #
bleu_scorer = BleuScore()
async def BLUE(inputs:dict, outputs:dict, reference_outputs:dict):
    result = await bleu_scorer.ascore(
        reference = reference_outputs['expected_output'],
        response = outputs['answer'] 
    )

    return result.value

# --------------------------------------------------------------------------- #
#                               RougeL       
# --------------------------------------------------------------------------- #
rouge_scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")
async def rougeL(inputs:dict, outputs:dict, reference_outputs:dict):
    result = await rouge_scorer.ascore(
        reference = reference_outputs['expected_output'],
        response = outputs['answer'] 
    )

    return result.value