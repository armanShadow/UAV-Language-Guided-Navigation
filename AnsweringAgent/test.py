import pandas as pd
from datasets import Dataset
from langchain_community.document_loaders import JSONLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch

print("Torch version:", torch.__version__)

# Define file paths
TRAIN_PATH = "../annotations/train_data.json"
VAL_PATH = "../annotations/val_seen_data.json"


# Load data function
def load_data(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema=".[].instructions",
        text_content=False,
    )

    raw_data = loader.load()
    data = []

    # Extract instructions and questions
    instructions = []
    questions = []
    for d in raw_data:
        ins_index = d.page_content.index("[INS]")
        instructions.append(d.page_content[ins_index + 5:])
        questions.append(d.page_content[5:ins_index])

    # Pair instructions with the next question
    for i in range(len(instructions)):
        if i < len(questions) - 1:
            # Pair with the next question
            data.append({
                "instruction": instructions[i],
                "question": questions[i + 1] if questions[i + 1].strip() else "No question"
            })
        else:
            # Last instruction gets "No question" by default
            data.append({
                "instruction": instructions[i],
                "question": "No question"
            })

    return pd.DataFrame(data)


# Load datasets
train_df = load_data(TRAIN_PATH)
val_df = load_data(VAL_PATH)


# Balance the dataset
def balance_data(df):
    """Balance 'No question' and valid question samples."""
    no_question_df = df[df["question"] == "No question"]
    question_df = df[df["question"] != "No question"]

    # Keep only a subset of "No question" examples
    # Combine and shuffle the datasets
    balanced_df = pd.concat([no_question_df[:int(len(no_question_df)/3)], question_df], ignore_index=True).sample(frac=1, random_state=42)
    return balanced_df


train_df = balance_data(train_df)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Preprocess function
def preprocess_function(examples):
    """Tokenize the input and output texts."""
    input_texts = ["generate question for the following instruction: " + inst for inst in examples["instruction"]]
    target_texts = examples["question"]

    # Tokenize input and target
    model_inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_texts, max_length=50, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize datasets
train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("fine-tuned-t5-model")
tokenizer.save_pretrained("fine-tuned-t5-model")


# Test the model with a sample input
def test_model(sample_instruction):
    """Generate a question for a given instruction."""
    inputs = tokenizer(
        f"generate question for the following instruction: {sample_instruction}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
