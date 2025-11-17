# Evlation Metrics for LLM and Agentic RAG

## Non-LLM Metrics
### EM
EM or Exact match mesures if the predicted / Genrated output (Afte Normallizaion: lowercasing, removing stop words, extra space,etc) actually matches the truth

Exact Match (EM):

$$
EM = 
\begin{cases}
1 & \text{if prediction = ground truth} \\
0 & \text{otherwise}
\end{cases}
$$

EM matches if the generated is exact the same of actual value but since our RAG/ llm can produce entirelly different set of tokens though aligned with actual answer. hence it should not be used for our tasks

<code>Llm_phase: NO USAGE, Agentic_phase: NO USAGE </code>

### F1

F1 score is harmonic mean of Precision and recall which is the harmonic mean of how much predict token that are correct and how much gold tokens that the model predicted


Precision and Recall:

$$
Precision = 
\frac{\text{count of overlapping tokens}}
{\text{count of predicted tokens}}
$$

$$
Recall = 
\frac{\text{count of overlapping tokens}}
{\text{count of ground truth tokens}}
$$



F1 Score:

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

Evaluating on the basis of overlapping tokens may give some idea about model performance when it is to be compared token level but when it would required to understand generated answers this would fail to trace the capability of such systems

<code>Llm_phase: OPTIONAL, Agentic_Phase: NO USAGE</code>

### Rouge

ROUGE or Recall-Oriented Understudy of Gisting Evaluation is typically used for evaluating Summerization, long answer generation or any generation task. It measures overlap between generated text and reference text at:
- n-grams
- subsequences
- longest common subsequence

ROUGE-1 : Measures content similarity.
$$
ROUGE\text{-}1 = 
\frac{\text{Overlapping unigrams}}{\text{Total unigrams in reference}}
$$

ROUGE-2 : Measures fluency/phrase similarity.

$$
ROUGE\text{-}2 = 
\frac{\text{Overlapping bigrams}}{\text{Total bigrams in reference}}
$$


ROUGE-L (Longest Common Subsequence Recall)
$$
ROUGE\text{-}L = 
\frac{LCS(\text{prediction},\ \text{reference})}{\text{Length of reference}}
$$

Rouge is recall-oreinted which is, How much of information of the refrence (actual) does model reproduce.

Such capability can be used for mesuring n-gram level overlapping, though it would still be non-capable to evaluate on the basis of sementic meaning of generated text and refrence answer.

<code>Llm_phase: OPTIONAL,
Agentic_Phase: NO USAGE</code>

### BERTScore
BERTScore uses Contextual embeddings from a language models to compute sementic similarity no just ngrams overlap.

It matches each token in the prediction to the most similar token in the refernce using cosing similarity. Which is if meaning is preserved in generated text, BERTScore is high

Token-Level cosing matching (Recall)
$$
Recall = 
\frac{1}{|R|}
\sum_{r \in R}
\max_{p \in P}
\cos \left( E(r), E(p) \right)
$$

Token-Level Cosing matching (Precision)

$$
Precision = 
\frac{1}{|P|}
\sum_{p \in P}
\max_{r \in R}
\cos \left( E(p), E(r) \right)
$$

BERTScore F1

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

Measuruing token level cosine similarity would actually tell us if the word with same meaning appeared in the generated text if though the actual word had not. Hence this should be used in both of evaluation: LLM_Phase and Agentic_Phase

<code>Llm_Phase: USAGE, Agentic_Phase: USAGE</code>

### Recall@K
Recall@K measure how many of the relvant documents were successfully retrieved in the top K results.

$$
\text{Recall@K} = \frac{| R \cap \text{retrieved@K} |}{|R|}
$$

Since this is directly aligned with retreival, it should be only used in Agentic rag evaluation

<code>Llm_phase: NO USAGE, Agentic_phase: USAGE</code>

### Precision@K
This measure > of the top k retrived documents, how many were actually relvant

$$
\text{Precision@K} = \frac{| R \cap \text{retrieved@K} |}{K}
$$

Here also, the it is direcly assoiated with retreival, hence it should only be used in Agentic rag evaluation.

<code>Llm_phase: NO USAGE, Agentic_phase: USAGE</code>

### nDCG@K

nDCG@K or Normallized Discounted Commulative Gain evaluates how well-ranked the retreiveed results are. the higher the score for relevant documetns appearing earlier.

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

$$
\text{IDCG@K} = \sum_{i=1}^{K} \frac{rel^{*}_i}{\log_2(i+1)}
$$

$$
\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

This metric would be used to evaluate re-ranking step after retreival and hence would only be used in Agentic rag phase.

<code> Llm_phase: NO USAGE, Agentic_phase: USAGE</code>

### MRR / MRR@K
Mean Reciprocal Rank or MRR evaluates the retrival process on the basis of the fact that only the first relvant result matters.

But Why would ranked context matters for genratio even where the top document in context matter most ?

- LLM have context limit which we could only fit with few retreived chunks and but this may reduce the chances of most relvent chunk to be inside this limitited context window and hence the llm wouldn't be able to generate most relvent answer

- LLMs have positional bias, which causes them to pay more attention to earlier context and hence the earlier token gets stronger attention and more influence on the final answer.

Reciprocal Rank for a single query

$$
RR_i = \frac{1}{\text{rank}_i}
$$

$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} RR_i
$$

$$
\text{MRR@K} = \frac{1}{N} \sum_{i=1}^{N}
\begin{cases}
\frac{1}{\text{rank}_i}, & \text{if } \text{rank}_i \le K \\
0, & \text{otherwise}
\end{cases}
$$


For each query, it takes the rank of the first relevant result, invert it and then average across all the queries.

<code>Llm_phase: NO USAGE, Agentic Phase: USAGE</code>

## LLM Metrics
### Correctness

<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### Groundness
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### Hallucinations
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### Faithfulness
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### relevence
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### Coherence
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

## System Metrics
### Latency
<code>Llm_phase: USAGE, Agentic Phase: USAGE</code>

### Memory
<code>Llm_phase: USAGE, Agentic Phase: NO USAGE</code>

### Throughput
<code>Llm_phase: USAGE, Agentic Phase: NO USAGE</code>