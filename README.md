# ResearchGemma-RAG


# Problem Statement
**Build a Research-Aware RAG system that retrieves, synthesizes, and summarizes scientific literature; provides citation-backed answers and concise multi-paper syntheses; and minimizes hallucinations for reproducible research support.**

 - Inputs : Free-text research question or request (e.g., “Summarize latest methods for X”, “Compare results across papers Y and Z”, “Find datasets used for task T”).

 - Outputs: Short synthesized answer (≤ 300 words) or structured summary (abstract + methods + key results), 1–5 cited snippets (paper title, section, paragraph, link/DOI), Confidence score and a short evidence trace listing the supporting papers and sections.

 - Success criteria: Retreive relvent papers on held-out queries, cited snippits supports the claim (citation consistency), Less hallucination rate

---