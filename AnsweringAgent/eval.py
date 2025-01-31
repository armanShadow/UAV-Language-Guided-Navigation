import evaluate
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.document_loaders import JSONLoader
import pandas as pd


# Load your fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("fine-tuned-t5-model")
tokenizer = T5Tokenizer.from_pretrained("fine-tuned-t5-model")


# Define file paths
TRAIN_PATH = "../annotations/train_data.json"
VAL_PATH = "../annotations/val_seen_data.json"
TEST_PATH = "../annotations/test_unseen_data.json"


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
test_df = load_data(TEST_PATH)

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


# Test on a sample instruction
sample_instruction = test_df.iloc[2]["instruction"]
expected_question = test_df.iloc[2]["question"]

generated_question = test_model(sample_instruction)
print("Input Instruction:", sample_instruction)
print("Expected Question:", expected_question)
print("Generated Question:", generated_question)


def evaluate_result(result):
    predictions = result.predictions
    label_ids = result.label_ids

    # Load the BLEU and ROUGE metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # Decode predictions and labels to text for text generation tasks
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in label_ids]

    # Calculate BLEU and ROUGE
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    print("BLEU:", bleu_result["bleu"])
    print("ROUGE-1:", rouge_result["rouge1"].mid.fmeasure)
