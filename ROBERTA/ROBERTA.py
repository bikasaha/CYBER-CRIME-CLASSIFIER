import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.functional import softmax
import json
import numpy as np

# Load data
train_df = pd.read_csv("TRAIN_AUG_PROCESSED.csv")
test_df = pd.read_csv("TEST_PROCESSED.csv")

print(train_df['category'].nunique())
print(test_df['category'].nunique())

# Assuming `df` is your DataFrame and `category` is the label column
rare_classes = train_df['category'].value_counts()[train_df['category'].value_counts() == 1].index

# Filter out rows with these rare classes in both train and test sets
train_df = train_df[~train_df['category'].isin(rare_classes)]
test_df = test_df[~test_df['category'].isin(rare_classes)]

print(train_df['category'].nunique())
print(test_df['category'].nunique())

# Print the number of unique classes to confirm
print("Unique classes in train after removal:", train_df['category'].nunique())
print("Unique classes in test after removal:", test_df['category'].nunique())

# Data Preprocessing
train_df.drop(columns=['sub_category'], inplace=True, errors='ignore')
test_df.drop(columns=['sub_category'], inplace=True, errors='ignore')
train_df['crimeaditionalinfo'].fillna('', inplace=True)
test_df['crimeaditionalinfo'].fillna('', inplace=True)

# # Standardize the labels
# standardized_labels = {
#     'Any Other Cyber Crime': 'Other Cyber Crime',
#     'Child Pornography CPChild Sexual Abuse Material CSAM': 'Child Abuse Material',
#     'Cryptocurrency Crime': 'Cryptocurrency Crime',
#     'Cyber Attack/ Dependent Crimes': 'Cyber Attack/Dependent Crimes',
#     'Cyber Terrorism': 'Cyber Terrorism',
#     'Hacking  Damage to computercomputer system etc': 'Hacking/Damage',
#     'Online Cyber Trafficking': 'Cyber Trafficking',
#     'Online Financial Fraud': 'Financial Fraud',
#     'Online Gambling  Betting': 'Gambling/Betting',
#     'Online and Social Media Related Crime': 'Social Media Crime',
#     'Ransomware': 'Ransomware',
#     'RapeGang Rape RGRSexually Abusive Content': 'Rape or Sexual Abuse Content',
#     'Report Unlawful Content': 'Unlawful Content Report',
#     'Sexually Explicit Act': 'Sexually Explicit Content',
#     'Sexually Obscene material': 'Sexually Obscene Content'
# }
# train_df['category'] = train_df['category'].replace(standardized_labels)
# test_df['category'] = test_df['category'].replace(standardized_labels)

train_df['crimeaditionalinfo'] = train_df['crimeaditionalinfo'].astype(str)
test_df['crimeaditionalinfo'] = test_df['crimeaditionalinfo'].astype(str)

# Encode labels
label_encoder = LabelEncoder()
train_df['category'] = label_encoder.fit_transform(train_df['category'])
print(test_df['category'].nunique())
# print(test_df['category'].unique())
test_df = test_df[test_df['category'].isin(label_encoder.classes_)]
print(test_df['category'].nunique())
test_df['category'] = label_encoder.transform(test_df['category'])

print(train_df['category'].nunique())
print(test_df['category'].nunique())
# print(test_df['category'].nunique())
# print(test_df)


# Split training data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['crimeaditionalinfo'].tolist(), 
    train_df['category'].tolist(), 
    test_size=0.2, 
    stratify=train_df['category']
)

# Define custom Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

# Tokenizer and Model for RoBERTA
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Create datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
test_texts = test_df['crimeaditionalinfo'].tolist()
test_labels = test_df['category'].tolist()
test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

# Directory to save the best model
best_model_path = "./BEST"

# Training arguments with early stopping
training_args = TrainingArguments(
    output_dir=best_model_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Disable automatic saving by Trainer
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,  # Disable auto-loading to manually load the best model
    logging_dir='./logs'
)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probabilities = softmax(torch.tensor(pred.predictions), dim=1).numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)

    # Define the full set of classes
    all_classes = list(range(train_df['category'].nunique()))  # All classes expected in the model (0 to 14)

    # Adjust `y_pred_probs` to ensure it includes probabilities for all classes
    # Create a zero matrix for the full set of classes
    adjusted_y_pred_probs = np.zeros((len(probabilities), len(all_classes)))

    # Fill the adjusted array with existing probabilities
    # Assume `y_pred_probs` columns correspond to classes in `y_true`
    present_classes = np.unique(labels)
    for i, class_idx in enumerate(present_classes):
        adjusted_y_pred_probs[:, class_idx] = probabilities[:, i]

    # Now compute log_loss with the adjusted probabilities
    log_loss_val = log_loss(labels, adjusted_y_pred_probs, labels=all_classes)

    balanced_acc = balanced_accuracy_score(labels, preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'log_loss': log_loss_val,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': cm.tolist()
    }

# Track the best F1 score
best_f1_score = 0.0
epoch_metrics = []

# Define a custom callback for saving the model and storing metrics each epoch
class SaveMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_count = 0  # Initialize epoch counter

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_count += 1
        print(f"Starting Epoch {self.epoch_count}...")
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        global best_f1_score
        if metrics:
            epoch_metrics.append(metrics)  # Store metrics for each epoch
            
            # Save best model based on F1 score
            if metrics.get("eval_f1", 0) > best_f1_score:
                best_f1_score = metrics["eval_f1"]
                trainer.save_model(best_model_path)
                print(f"Model saved with improved F1 score: {best_f1_score}")

# Initialize Trainer with EarlyStoppingCallback and SaveBestModelCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), SaveMetricsCallback()]
)

# Train the model with early stopping
trainer.train()

# last_model = './LAST'

# # Explicitly save the best model after training
# trainer.save_model(last_model)

# Save epoch metrics to JSON
with open("epoch_metrics.json", "w") as f:
    json.dump(epoch_metrics, f, indent=4)

# Save the label encoder for decoding predictions later
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
with open("label_mapping.json", "w") as label_file:
    json.dump(label_mapping, label_file)

# Load the best model for evaluation
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Evaluate on the test set
test_metrics = trainer.evaluate()
with open("test_metrics.json", "w") as test_file:
    json.dump(test_metrics, test_file)

# Print the test metrics
print("\nTest Evaluation Metrics:", test_metrics)
