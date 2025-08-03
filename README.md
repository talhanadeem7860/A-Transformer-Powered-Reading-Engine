Project Overview

This project implements a sophisticated Question Answering (QA) system that demonstrates true reading comprehension. Instead of just searching for keywords, this application uses a fine-tuned Transformer model (DistilBERT) to read a given passage of text and find the precise answer to a user's question.

This repository showcases a complete, modern Natural Language Processing (NLP) pipeline, from data processing and model fine-tuning to intelligent inference on custom text.

Methodology

The project is executed in two main phases:

Model Fine-Tuning:

The process begins by loading a general-purpose, pre-trained DistilBERT model and its tokenizer from the Hugging Face library. This model is then specialized for question answering by fine-tuning it on the Stanford Question Answering Dataset (SQuAD v2). The training process, managed by the Hugging Face Trainer API, adjusts the model's parameters to master the task of locating answers within a text. The resulting fine-tuned model is saved locally.

Inference and Question Answering:

Once trained, an inference pipeline loads the specialized model. This pipeline is designed to accept a new, custom paragraph of text (the context) and a user-provided question. It processes this input and uses the model to predict the start and end positions of the most likely answer within the context, effectively performing reading comprehension to extract the correct information.
