# For List of All Evaluation metrics, Check: https://github.com/Se00n00/ResearchGemma-RAG/blob/main/docs/evaluation_metrics.md
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

LLM = os.getenv("LLM")
BASE_URL = os.getenv("BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class MetricScore(TypedDict):
    score: Annotated[int, ..., "Score <1-10>"]

judge = ChatOpenAI(
    model = LLM,
    api_key = GROQ_API_KEY,
    base_url = BASE_URL,
    streaming = True
).with_structured_output(MetricScore, method="json_schema", strict=True)



# --------------------------------------------------------------------------- #
#                               CORRECTNESS
# --------------------------------------------------------------------------- #

correctness_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluator. Score the correctness of the generated response "
     "compared to the reference answer. Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "USER QUESTION:\n{question}\n\n"
     "GENERATED ANSWER:\n{answer}\n\n"
     "REFERENCE ANSWER:\n{reference}\n\n"
     "Return JSON only."
    )
])

def correctness(inputs: dict, outputs: dict) -> int:
    msgs = correctness_prompt.format_messages(
        question=inputs["question"],
        answer=outputs["answer"],
        reference=inputs["reference_answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]



# --------------------------------------------------------------------------- #
#                               GROUNDNESS
# --------------------------------------------------------------------------- #

groundness_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate how grounded the answer is in the retrieved documents.\n"
     "Use ONLY the docs, no world knowledge.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "RETRIEVED DOCS:\n{docs}\n\n"
     "GENERATED ANSWER:\n{answer}"
    )
])

def groundness(inputs: dict, outputs: dict) -> int:
    docs = "\n\n".join(d.page_content for d in outputs["documents"])
    msgs = groundness_prompt.format_messages(
        docs=docs,
        answer=outputs["answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]


# --------------------------------------------------------------------------- #
#                               HALLUCINATION
# --------------------------------------------------------------------------- #

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate intrinsic hallucination. Check if answer contradicts real-world facts.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "QUESTION:\n{question}\n\n"
     "GENERATED ANSWER:\n{answer}"
    )
])

def hallucination_llm(inputs: dict, outputs: dict) -> int:
    msgs = hallucination_prompt.format_messages(
        question=inputs["question"],
        answer=outputs["answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]


# --------------------------------------------------------------------------- #
#                               FAITHFULNESS
# --------------------------------------------------------------------------- #

faithfulness_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate whether the answer relies ONLY on retrieved evidence.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "RETRIEVED DOCS:\n{docs}\n\n"
     "GENERATED ANSWER:\n{answer}"
    )
])

def faithfulness(inputs: dict, outputs: dict) -> int:
    docs = "\n\n".join(d.page_content for d in outputs["documents"])
    msgs = faithfulness_prompt.format_messages(
        docs=docs,
        answer=outputs["answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return 


# --------------------------------------------------------------------------- #
#                               RELEVANCE
# --------------------------------------------------------------------------- #

relevance_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate how well the answer addresses the user's question.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "QUESTION:\n{question}\n\n"
     "ANSWER:\n{answer}"
    )
])

def relevance(inputs: dict, outputs: dict) -> int:
    msgs = relevance_prompt.format_messages(
        question=inputs["question"],
        answer=outputs["answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]



# --------------------------------------------------------------------------- #
#                           RETRIEVAL RELEVANCE
# --------------------------------------------------------------------------- #

retrieval_relevance_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate how relevant the retrieved documents are for the user's query.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "QUESTION:\n{question}\n\n"
     "RETRIEVED DOCS:\n{docs}"
    )
])

def retreival_relevance(inputs: dict, outputs: dict) -> int:
    docs = "\n\n".join(d.page_content for d in outputs["documents"])
    msgs = retrieval_relevance_prompt.format_messages(
        question=inputs["question"],
        docs=docs
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]



# --------------------------------------------------------------------------- #
#                               COHERENCE
# --------------------------------------------------------------------------- #

coherence_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate clarity, structure, and coherence.\n"
     "Return ONLY JSON: 'score': <1-10>"
    ),

    ("human",
     "GENERATED ANSWER:\n{answer}"
    )
])

def coherence(inputs: dict, outputs: dict) -> int:
    msgs = coherence_prompt.format_messages(
        answer=outputs["answer"]
    )
    res: MetricScore = judge.invoke(msgs)
    return res["score"]