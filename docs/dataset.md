# Dataset

For Evaluating LLM and Agentic-Rag i needed to find a golden dataset that meet the following criteria:

 - Able to evaluate both llm and agentic phase for same task: Research Paper QA
 - Contains data that is enriched enough to be evaluated on key metrics like hallucinations, coherence and other llm metrics. [Metrics](evaluation_metrics.md)

Usually Though Finding such QA dataset for this task would be tidious but since rag had been a popular for over few years there exists numberous of dataset on open source platforms like [Huggingface](https://huggingface.co/datasets), where i find few dataset that might do my job:

## Squad by rajpurkar

**Description**
```
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
```

**Example Data**
```
{
    "answers": {
        "answer_start": [1],
        "text": ["This is a test text"]
    },
    "context": "This is a test context.",
    "id": "1",
    "question": "Is this a test?",
    "title": "train test"
}

```

This dataset had what i required : a QA pair with context but had following issues:
 - The answers were very short which are usually suitable for extractive RAG.
 - due to short answers i couldn't evaluate on complex metrics like coherence, groundness, etc


## Qasper by allenai

**Description**

```
QASPER is a dataset for question answering on scientific research papers. It consists of 5,049 questions over 1,585 Natural Language Processing papers. Each question is written by an NLP practitioner who read only the title and abstract of the corresponding paper, and the question seeks information present in the full text. The questions are then answered by a separate set of NLP practitioners who also provide supporting evidence to answers.
```

**Example Data**

```
{
  'id': "Paper ID (string)",
  'title': "Paper Title",
  'abstract': "paper abstract ...",
  'full_text': {
      'paragraphs':[["section1_paragraph1_text","section1_paragraph2_text",...],["section2_paragraph1_text","section2_paragraph2_text",...]],
      'section_name':["section1_title","section2_title"],...},
  'qas': {
  'answers':[{
      'annotation_id': ["q1_answer1_annotation_id","q1_answer2_annotation_id"]
      'answer': [{
          'unanswerable':False,
          'extractive_spans':["q1_answer1_extractive_span1","q1_answer1_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer1",
          'evidence':["q1_answer1_evidence1","q1_answer1_evidence2",..],
          'highlighted_evidence':["q1_answer1_highlighted_evidence1","q1_answer1_highlighted_evidence2",..]
          },
          {
          'unanswerable':False,
          'extractive_spans':["q1_answer2_extractive_span1","q1_answer2_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer2",
          'evidence':["q1_answer2_evidence1","q1_answer2_evidence2",..],
          'highlighted_evidence':["q1_answer2_highlighted_evidence1","q1_answer2_highlighted_evidence2",..]
          }],
      'worker_id':["q1_answer1_worker_id","q1_answer2_worker_id"]
      },{...["question2's answers"]..},{...["question3's answers"]..}],
  'question':["question1","question2","question3"...],
  'question_id':["question1_id","question2_id","question3_id"...],
  'question_writer':["question1_writer_id","question2_writer_id","question3_writer_id"...],
  'nlp_background':["question1_writer_nlp_background","question2_writer_nlp_background",...],
  'topic_background':["question1_writer_topic_background","question2_writer_topic_background",...],
  'paper_read': ["question1_writer_paper_read_status","question2_writer_paper_read_status",...],
  'search_query':["question1_search_query","question2_search_query","question3_search_query"...],
  }
}

```

This is exactily what i needed for my project:
- Dataset is aligned to my task: Contains QAs on the basis of content of actual research papers
- Had evidence and highlighted_evidence which would be good for evaluating retreival and this agentic rag
- Had long and quality answer that i can evaluate my agent as well as my custom llms on desired metrics

## Dataset Prepration

**For LLM_Phase**

*For llm_phase the evlauation you be on following llm metrics: Correctness, Groundness, Hallucinations, faithfulness, relevence, coherence.*

*And also on system metrics such as latency, memory and throughput*

**For Agentic_Phase**

*For Agentic phase the evaluation would be on llm_metrics and non_llm_metrics such as EM, F1, Rouge and BERT_score.*

*Including rag-specific metrics: recall@k, precision@k, nDCG@k, MRR*

### Final Dataset Format:

<code> [Qasper_papers](../data/qasper_paper.jsonl) </code> 
```
{
    paper_id : paper_context
}
```

<code> [Qasper_qa](../data/qasper_qa.jsonl) </code> : Langsmith format 
```
{
    input: str,
    expected_output: str,
    meta_data: {
        evidence: [],
        highlighted_evidence: [],
        context_id: paper_id
    }
}
```