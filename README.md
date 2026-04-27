# RAG Psychiatry QA
An RAG system for Psychiatric Question Answering. Submitted for University of New Haven DSCI 6004 - Natural Language Processing.

### Summary
- this project implements and compares the performance of three major LLMs (LLaMa, Mistral, Gemma) for an RAG system specifically for answering questions related to psychiatry.
- this model ingests five authoritative documents in the realm of psychiatry (DSM-5, ICD, WHO mental health guidance, open-source psychiatry textbooks, APA guidance...), using PubMedBERT embeddings.
- we use 17 psychiatric questions (spanning questions related to diagnosis, treatment, cultural considerations in psychiatry, symptom lists...) to evaluate the performance of the RAG
- we evaluate performance using BLEU/ROUGE, recall@k, precision@k and semantic similarity (cosine similarity)
- Overall, models performed similarly, and were able to provide helpful and clinically accurate responses to psychiatric questions
- BLEU scores indicated that models (particularly LLaMa) were not simply regurgitating content provided from the context, but reframing answers.

### Code 
- All coding was done in Google Colab to leverage the Nvidia GPU processors and speed up model implementation.
- HuggingFace was used for model access. To replicate this project, you will need your own hugging face authentification. https://huggingface.co/
- [RAG Implementation and Evaluation Jupyter Notebook](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/NLP_RAG_Project%20(1).md) (markdown)
- [Google Drive Link to Models, Embeddings](https://drive.google.com/drive/folders/1YKQrMhvLU__tTl7sk96ALz6qiw7d7C6e?usp=sharing)

### Results
- Semantic Similarity Heatmap: Gemma vs Mistral:

![Semantic Similarity Heatmap Gemma vs Mistral](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/output_102_0.png)

- BLEU vs ROUGE (all 3 models):
![BLEU vs ROUGE](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/output_82_1.png)

- Precision, Recall and F1 (BERTscore) - uses BERT (another llm) to compute the clinical precision and recall of the answers:

![Precision, Recall and F1](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/output_116_0.png)

- BERTscore F1 per question:

![F1 per question](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/output_117_0.png)

- Precision and Recall @ K:

![Precision and Recall @ K](https://github.com/ellenmartin11/RAG-Psychiatry-QA/blob/main/analysis%20and%20results/output_127_0.png)
