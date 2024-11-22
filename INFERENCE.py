import nltk
nltk.download('punkt_tab')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
from torch.nn.functional import softmax
import spacy

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the tokenizer and the trained model
model_name = 'l3cube-pune/hing-roberta'  # Update with the model you trained
best_model_path = './MODEL'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
model.eval()  # Set the model to evaluation mode

# Load the label mapping
with open("label_mapping.json", "r") as label_file:
    label_mapping = json.load(label_file)
    label_mapping = {int(k): v for k, v in label_mapping.items()}

# Define a function for preprocessing and inference
def predict(texts, tokenizer, model, max_length=128):
    """
    Perform inference on a list of texts.
    
    Args:
        texts (list): List of strings (text data).
        tokenizer (AutoTokenizer): Pretrained tokenizer.
        model (AutoModelForSequenceClassification): Trained model.
        max_length (int): Maximum sequence length for tokenization.
        
    Returns:
        list: Predictions and corresponding probabilities.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
    
    return predictions


# Step 2: Text Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s<>]', '', text)  # Remove special characters, keeping placeholders
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Function to mask only personal/sensitive information
def mask_sensitive_entities(text):
    # Apply spaCy's NER
    doc = nlp(text)
    
    # Define sensitive entity types to be masked
    sensitive_entity_types = {
        "PERSON",       # Names of people
        "EMAIL",        # Email addresses
        "PHONE",        # Phone numbers (not included by default; add with custom model)
        "CARDINAL",     # Numeric identifiers (may include account or card numbers)
        "MONEY",        # Monetary values
        "DATE",         # Dates (important for privacy)
        "ADDRESS"       # Addresses (custom entities, requires training or patterns)
    }
    
    # Replace only sensitive entities with their label
    for ent in doc.ents:
        if ent.label_ in sensitive_entity_types:
            text = text.replace(ent.text, f"<{ent.label_}>")
    
    return text

# Define a function to preprocess text
def remove_stopwords(text):
    """
    Remove stopwords from the input text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Preprocessed text.
    """
    return ' '.join(word for word in word_tokenize(text) if word.lower() not in stop_words)

def lemmatize_text(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())


# Example texts for inference
example_text = "My salary of 800 Rs was supposed to be credited to my account by Jan 1st, but due to some issue with the bank, it was not credited. So, I searched online and called a customer care number. The person I spoke to asked me to message my ICICI account and debit card number. Additionally, he requested me to download the AnyDesk app on my mobile. Later, he asked me to provide the last digits of my mobile number and mentioned that this was required for my ICICI ATM card. Subsequently, he instructed me to transfer an amount twice—once from my bank and once from PhonePe. The IFSC code he provided for the transfer was BKIDXXXXXXX. In total, an amount of 50000 Rs was debited from my account."

texts_to_predict = clean_text(example_text)

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")
print("Loaded spacy.")

texts_to_predict = mask_sensitive_entities(texts_to_predict)
# print(texts_to_predict)

# Define stopwords and retain contextually relevant ones
stop_words = set(stopwords.words('english')) - {'not', 'no', 'because'}
texts_to_predict = remove_stopwords(texts_to_predict)

# Step 5: Lemmatization
lemmatizer = WordNetLemmatizer()
texts_to_predict = lemmatize_text(texts_to_predict)

# Perform inference
predictions = predict(texts_to_predict, tokenizer, model)

# Map predictions back to labels
predicted_labels = [label_mapping[pred] for pred in predictions]

print("-" * 30)
print(f"Text: {example_text}")
print("-" * 30)
print(f"Predicted Label: {predicted_labels}")
print("-" * 30)

# # Print results
# for text, label in zip(texts_to_predict, predicted_labels):
#     print(f"Text: {text}")
#     print(f"Predicted Label: {label}")
#     print("-" * 30)
