from transformers import pipeline


print("\n[INFO] Loading fine-tuned model for inference...")

qa_pipeline = pipeline(
    "question-answering",
    model="./my_qa_model",
    tokenizer="./my_qa_model"
)

context = """
The Amazon rainforest, covering much of northwestern Brazil and extending into Colombia, 
Peru and other South American countries, is the world's largest tropical rainforest, 
famed for its biodiversity. It’s crisscrossed by thousands of rivers, including the powerful Amazon.
"""

questions = [
    "Which is the world's largest tropical rainforest?",
    "Which river flows through the rainforest?",
    "What is the capital of France?" 
]

print("\n--- Running Inference ---")
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
 
    if result['score'] > 0.1:
        print(f"Answer: '{result['answer']}' (Score: {result['score']:.4f})")
    else:
        print("Answer: I don't know.")
    print("-" * 20)