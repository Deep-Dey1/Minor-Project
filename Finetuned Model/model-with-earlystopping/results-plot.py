from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, Dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the fine-tuned model checkpoint
checkpoint_dir = "C:/Users/AIC MUJ W2/Deep-Iot-C-LLm/Mistral-Chatbot-v3/mistral-finetuned-alpaca/checkpoint-500"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",  # Original base model
    device_map="auto",  # Automatically maps the model to available devices (e.g., GPU)
    torch_dtype=torch.float16  # Use FP16 for faster inference (optional)
)

# Load the fine-tuned adapter weights from the checkpoint
model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# Load the tokenizer from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# Load the validation dataset
data = load_dataset("tatsu-lab/alpaca", split="train")
data_df = data.to_pandas()
val_df = data_df[4000:4100]  # Use only 100 samples for validation

# Fix Pandas warning by using .loc for assignment
val_df.loc[:, "text"] = val_df[["input", "instruction", "output"]].apply(
    lambda x: "###Human: " + x["instruction"] + " " + x["input"] + " ###Assistant: " + x["output"], axis=1
)
val_data = Dataset.from_pandas(val_df)

# Move model to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate predictions and collect ground truth labels
predictions = []
labels = []

for example in val_data:
    inputs = tokenizer(example["text"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Adjust max_new_tokens as needed
            eos_token_id=tokenizer.eos_token_id,  # Ensure the model stops at the EOS token
            pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
            no_repeat_ngram_size=2,  # Prevent repetition of n-grams
            do_sample=True,  # Enable sampling for more diverse outputs
            top_p=0.9,  # Nucleus sampling
            temperature=0.7,  # Control randomness
        )
    predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    labels.append(example["output"])  # Ground truth

# Compute BLEU scores
bleu_scores = []
for pred, label in zip(predictions, labels):
    # Tokenize the prediction and label
    pred_tokens = pred.split()
    label_tokens = label.split()
    # Compute BLEU score
    bleu_score = sentence_bleu([label_tokens], pred_tokens)
    bleu_scores.append(bleu_score)

# Compute average BLEU score
avg_bleu_score = np.mean(bleu_scores)
print(f"Average BLEU Score: {avg_bleu_score}")

# Compute ROUGE scores
rouge = Rouge()
rouge_scores = rouge.get_scores(predictions, labels, avg=True)

# Print ROUGE scores
print(f"ROUGE-1: {rouge_scores['rouge-1']['f']}")
print(f"ROUGE-2: {rouge_scores['rouge-2']['f']}")
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']}")

# Visualize the results
metrics = {
    "BLEU": avg_bleu_score,
    "ROUGE-1": rouge_scores["rouge-1"]["f"],
    "ROUGE-2": rouge_scores["rouge-2"]["f"],
    "ROUGE-L": rouge_scores["rouge-l"]["f"],
}

# Plot the metrics
plt.figure(figsize=(10, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title("Model Evaluation Metrics (Validation on 100 Samples)")
plt.ylabel("Score")
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.show()