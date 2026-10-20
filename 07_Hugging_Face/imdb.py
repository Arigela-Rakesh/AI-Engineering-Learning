import torch

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load IMDB dataset

dataset = load_dataset("stanfordnlp/imdb")

print("Dataset:", dataset)

print("\nFirst training example:")

print(dataset["train"][0])

# Load pretrained Tokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=128)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("\nTokenized example:")

print(tokenized_dataset["train"][0])

# Load Pretrained Classification Model

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create Label Mapping

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
# Inference Function


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        sentiment = id2label[predicted_class]
        return sentiment


text = "This movie was absolutely fantastic!"

result = predict_sentiment(text)

print("\nInput:")

print(text)

print("\nPrediction")

print(result)
