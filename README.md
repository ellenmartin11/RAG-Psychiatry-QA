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
- RAG Implementation and Evaluation Jupyter Notebook.
