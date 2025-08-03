# --- Data Preprocessing ---
print("[INFO] Preprocessing the dataset...")


max_length = 384 
doc_stride = 128 

def preprocess_function(examples):
    """
    Preprocesses the data by tokenizing and aligning labels.
    Handles both training and validation sets.
    """
    # Tokenize questions and contexts
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second", # Truncate the context, not the question
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        
      
        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            
            idx = context_start
            while idx <= context_end and offsets[idx][0] < start_char:
                idx += 1
            start_positions.append(idx)

            idx = context_end
            while idx >= context_start and offsets[idx][1] > end_char:
                idx -= 1
            end_positions.append(idx)
            
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Apply the preprocessing function to the entire dataset
tokenized_squad = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
print("[INFO] Preprocessing complete.")