# IMPORT LIBRARIES

# torch → core deep learning framework used to run the model (CPU/GPU operations)
import torch

# load_dataset → loads ready-made NLP datasets easily
from datasets import load_dataset

# Hugging Face tools:
# AutoTokenizer → converts raw text into numerical tokens that BERT understands
# AutoModelForSequenceClassification → pretrained BERT model with classification head
# TrainingArguments → defines training configuration
# Trainer → handles full training loop automatically
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# STEP 1: LOAD DATASET

# AG News dataset contains news articles categorized into 4 classes:
# 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech
# Dataset is already split into train and test
dataset = load_dataset("ag_news")


# STEP 2: LOAD TOKENIZER

# Tokenizer converts text → tokens (numbers)
# bert-base-uncased:
# - "base" → standard BERT model
# - "uncased" → converts all text to lowercase
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# STEP 3: TOKENIZATION FUNCTION

# Neural networks cannot understand text directly
# So we convert each sentence into token IDs + attention masks
def tokenize(example):

    return tokenizer(
        example["text"],

        # padding ensures all inputs have same length
        # this is required for batch processing
        padding="max_length",

        # truncation cuts long sentences to maximum allowed length
        truncation=True
    )


# Apply tokenization to entire dataset
# batched=True → processes multiple samples at once (faster)
dataset = dataset.map(tokenize, batched=True)


# STEP 4: PREPARE DATA FOR MODEL

# Rename label → labels
# Trainer expects label column to be named "labels"
dataset = dataset.rename_column("label", "labels")

# Convert dataset into PyTorch tensors
# input_ids → token numbers
# attention_mask → tells model which tokens are real vs padding
# labels → correct category
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# STEP 5: LOAD PRETRAINED MODEL

# This loads BERT with a classification head on top
# num_labels=4 → output layer has 4 neurons (one per class)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)


# STEP 6: TRAINING CONFIGURATION

# These parameters control how training happens
training_args = TrainingArguments(

    output_dir="./results",   # folder where model checkpoints are saved

    # small learning rate is important for pretrained models
    learning_rate=2e-5,

    # batch size → number of samples processed at once
    per_device_train_batch_size=16,

    # number of full passes through dataset
    num_train_epochs=1,

    logging_dir="./logs"      # stores training logs
)


# STEP 7: TRAINER

# Trainer handles:
# - forward pass
# - loss calculation
# - backpropagation
# - optimizer updates
# So we do not need to write training loop manually
trainer = Trainer(
    model=model,
    args=training_args,

    # Use smaller subset for faster training
    train_dataset=dataset["train"].select(range(2000)),
    eval_dataset=dataset["test"].select(range(500))
)


# STEP 8: TRAIN MODEL

# This automatically trains the model using above configuration
trainer.train()


# STEP 9: DEVICE SETUP

# Check if GPU is available
# GPU makes training and prediction faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to selected device
model.to(device)


# STEP 10: TEST WITH CUSTOM INPUT

# Example input sentence
text = "India won the T20 series"

# Convert text into model input format
inputs = tokenizer(
    text,
    return_tensors="pt",   # return PyTorch tensors
    truncation=True,
    padding=True
)

# Move inputs to same device as model (CPU/GPU)
inputs = {key: value.to(device) for key, value in inputs.items()}


# STEP 11: PREDICTION

# Pass input through model
outputs = model(**inputs)

# logits → raw scores for each class
# argmax → selects class with highest score
predicted_class = torch.argmax(outputs.logits).item()


# CLASS LABELS

labels = ["World", "Sports", "Business", "Sci/Tech"]

print("Text:", text)
print("Predicted Category:", labels[predicted_class])
