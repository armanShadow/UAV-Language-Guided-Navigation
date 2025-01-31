from langchain_community.document_loaders import JSONLoader
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5ForConditionalGeneration

import torch
print(torch.__version__)
print(torch.backends.mps.is_available())


def load_data(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema=".[].instructions",
        text_content=False,
    )

    raw_data = loader.load()
    data = []

    # set of instructions in the train data set
    instructions = []

    # set of questions in the train data set
    questions = []
    for d in raw_data:
        ins_index = d.page_content.index('[INS]')
        instructions.append(d.page_content[ins_index + 5:])
        questions.append(d.page_content[5:ins_index])

    # For the first task, given the instruction, we must ask the proper question
    for i in range(len(instructions)):
        if i != len(instructions) - 1:
            dialogue = {
                'instruction': instructions[i],
                'question': questions[i + 1]
            }
        else:
            # should handle the last element manually
            dialogue = {
                'instruction': instructions[i],
                'question': ' '
            }
        data.append(dialogue)

    return data



train_data = load_data('../annotations/train_data.json')

filtered_data = [d for d in train_data if d['question'].strip().lower() != ""]
no_question_data = [d for d in train_data if d['question'].strip().lower() == ""]

# Keep only a subset of "No question" examples
balanced_data = filtered_data + no_question_data[:int(len(filtered_data)/3)]

test_data = load_data('../annotations/test_unseen_data.json')
val_seen_data = load_data('../annotations/val_seen_data.json')
val_unseen_data = load_data('../annotations/val_unseen_data.json')

# Convert to DataFrame
train_df = pd.DataFrame(balanced_data, columns=['instruction', 'question'])
test_df = pd.DataFrame(test_data, columns=['instruction', 'question'])
val_seen_df = pd.DataFrame(val_seen_data, columns=['instruction', 'question'])
val_unseen_df = pd.DataFrame(val_unseen_data, columns=['instruction', 'question'])

# Create a Dataset object from the DataFrame
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_seen_dataset = Dataset.from_pandas(val_seen_df)
val_unseen_dataset = Dataset.from_pandas(val_unseen_df)

# Load the tokenizer and model
model_name = "t5-small"  # You can use t5-base or t5-large for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Prepare the dataset for the T5 model
def preprocess_function(examples):
    input_texts = ["generate question for the following instruction: " + inst for inst in examples['instruction']]

    # Handle empty target texts by replacing them with a placeholder
    target_texts = [q if q.strip() != "" else "No question" for q in examples['question']]

    # Tokenize the inputs
    model_inputs = tokenizer(input_texts, max_length=512, padding="max_length", truncation=True)

    # Tokenize the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, max_length=512, padding="max_length", truncation=True)

    # Assign labels
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Tokenize the dataset
train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
val_seen_tokenized_dataset = val_seen_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_seen_tokenized_dataset,
)

# Train the model
trainer.train()


# Save the model
trainer.save_model("fine-tuned-t5-model")
tokenizer.save_pretrained("fine-tuned-t5-model")