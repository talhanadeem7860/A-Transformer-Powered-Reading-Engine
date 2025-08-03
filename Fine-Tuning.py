# --- Fine-Tuning with Trainer API ---
print("[INFO] Setting up the Trainer...")

# Define the training arguments
args = TrainingArguments(
    output_dir="distilbert-finetuned-squadv2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3, # Can increase for better performance
    weight_decay=0.01,
    push_to_hub=False, # Set to True to upload to Hugging Face Hub
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
)


print("[INFO] Starting fine-tuning...")
trainer.train()
print("[INFO] Fine-tuning complete.")

# --- Save the fine-tuned model ---
print("[INFO] Saving the fine-tuned model locally...")
trainer.save_model("./my_qa_model")
print("Model saved to ./my_qa_model")