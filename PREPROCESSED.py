import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
import re


# Load your dataset
data = pd.read_csv("TRAIN_AUG.csv")
test_data = pd.read_csv("test.csv")

# Step 1: Remove Duplicates and Null Values
data.drop_duplicates(inplace=True)
data.dropna(subset=['crimeaditionalinfo'], inplace=True)
test_data.drop_duplicates(inplace=True)
test_data.dropna(subset=['crimeaditionalinfo'], inplace=True)
print("Removed NaN and Duplicates.")

# Step 2: Text Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s<>]', '', text)  # Remove special characters, keeping placeholders
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

data['crimeaditionalinfo'] = data['crimeaditionalinfo'].apply(clean_text)
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].apply(clean_text)
print("Cleaned Texts.")

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")
print("Loaded spacy.")

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

data['crimeaditionalinfo'] = data['crimeaditionalinfo'].apply(mask_sensitive_entities)
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].apply(mask_sensitive_entities)
print("Tagged using spacy.")

# Step 4: Stopword Removal
stop_words = set(stopwords.words('english')) - {'not', 'no', 'because'}  # Keeping contextually relevant stopwords
data['crimeaditionalinfo'] = data['crimeaditionalinfo'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words))
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words))
print("removed stopwords.")

# Step 5: Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

data['crimeaditionalinfo'] = data['crimeaditionalinfo'].apply(lemmatize_text)
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].apply(lemmatize_text)
print("Lemmatized.")

# Standardize the labels
standardized_labels = {
    'Any Other Cyber Crime': 'Other Cyber Crime',
    'Child Pornography CPChild Sexual Abuse Material CSAM': 'Child Abuse Material',
    'Cryptocurrency Crime': 'Cryptocurrency Crime',
    'Cyber Attack/ Dependent Crimes': 'Cyber Attack/Dependent Crimes',
    'Cyber Terrorism': 'Cyber Terrorism',
    'Hacking  Damage to computercomputer system etc': 'Hacking/Damage',
    'Online Cyber Trafficking': 'Cyber Trafficking',
    'Online Financial Fraud': 'Financial Fraud',
    'Online Gambling  Betting': 'Gambling/Betting',
    'Online and Social Media Related Crime': 'Social Media Crime',
    'Ransomware': 'Ransomware',
    'RapeGang Rape RGRSexually Abusive Content': 'Rape or Sexual Abuse Content',
    'Report Unlawful Content': 'Unlawful Content Report',
    'Sexually Explicit Act': 'Sexually Explicit Content',
    'Sexually Obscene material': 'Sexually Obscene Content'
}
data['category'] = data['category'].replace(standardized_labels)
test_data['category'] = test_data['category'].replace(standardized_labels)
print("Standardized Labels.")

data.to_csv("TRAIN_AUG_PROCESSED.csv",index=False)
test_data.to_csv("TEST_PROCESSED.csv",index=False)