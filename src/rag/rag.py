from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter


from typing import TypedDict
import chromadb
from transformers import pipeline
import os 
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class State(TypedDict):
  query: str
  context: list[str]
  answer: str

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="RAG")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", token = HF_TOKEN)

def run(prompt):
    output = pipe(prompt)
    return output


# -------------------------------------------- #
#                   Retrieve
# -------------------------------------------- #

def retreive(state:State):
  """
  query RAG using a local vector database
  """

  context = collection.query(
    query_texts = state['query'],
    n_results = 5,
    include = ["documents", "distances"]
  )['documents']

  return {"context":context}
# -------------------------------------------- #
#                   Upsert
# -------------------------------------------- #

def upsert(content: str, start_count:int):
  texts = text_splitter.split_text(content)
  collection.upsert(
      ids= [f"id{start_count + i}" for i in range(len(texts))],
      documents = texts
  )
  return len(texts)
# -------------------------------------------- #
#                   Generate
# -------------------------------------------- #


answering_prompt = """
  You are an assistant that answers the user's question using ONLY the provided context.
  if you don't know the answer say - i don't know the answer
"""

def answer(state:State):
  """
  generate the final answer using the LLM and retrieved documents
  """

  prompt = ChatPromptTemplate([
    ("system", answering_prompt),
    ("user", "Question: {question} \n\n Context: {context}"),
  ])

  query = prompt.invoke({"question":state['query'], "context":state['context']}).messages

  # Convert for gemma model
  queries = []
  for q in query:
      role = 'user' if q.type == 'human' else q.type

      queries.append({
          'role': role,
          'content': q.content
      })
  result = run(queries)

  return {"answer":result[0]['generated_text'][-1]['content']}
# -------------------------------------------- #
#                   RAG Pipeline
# -------------------------------------------- #
rag = (
    StateGraph(State)
    .add_sequence([retreive, answer])
    .add_edge(START, "retreive")
    .compile()
)