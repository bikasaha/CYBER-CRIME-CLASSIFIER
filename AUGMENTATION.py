from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import random
import csv
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

# Load BERT model and tokenizer for MLM
model_name = 'l3cube-pune/hing-roberta'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Move the model to the appropriate device
model = model.to(device) # This line ensures the model is on the correct device

# Set up the pipeline with the correct device
mlm_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)

# Load Sentence Transformer model for similarity calculation
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# Path for the intermediate CSV file
output_csv_path = 'TRAIN_AUG.csv'

# Initialize CSV file with headers
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['category', 'sub_category', 'crimeaditionalinfo', 'augmented_crimeaditionalinfo'])

# Function to augment text using Masked Language Model and filter by similarity
def augment_text(row, num_masks=30, num_augmentations=5, similarity_threshold=0.98):
    category, subcategory, text = row['category'], row['sub_category'], row['crimeaditionalinfo']
    original_embedding = similarity_model.encode(text, convert_to_tensor=True)
    
    words = text.split()
    max_masks = min(num_masks, len(words))  # Ensure num_masks does not exceed the number of words
    
    for _ in range(num_augmentations):
        if len(words) < 2:
            continue
        
        # Select words to mask
        masked_indices = random.sample(range(len(words)), max_masks)
        
        augmented_sentence = words[:]
        for i in masked_indices:
            masked_text = " ".join(
                [tokenizer.mask_token if idx == i else word for idx, word in enumerate(augmented_sentence)]
            )
            
            # Generate predictions for the masked token
            # Pass truncation and max_length to the tokenizer
            inputs = tokenizer(masked_text, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device) # Move inputs to the correct device
            
            if tokenizer.mask_token_id not in inputs.input_ids:
                print(f"No [MASK] token found in input: {masked_text}")
                continue  # Skip if [MASK] is missing
            
            masked_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            
            if masked_positions[0].size(0) == 0:
                print(f"No [MASK] token found in input: {masked_text}")
                continue
            
            masked_index = masked_positions[1][0]

            # Get model output
            with torch.no_grad():
                outputs = model(**inputs).logits
            
            # Process output
            masked_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
            predicted_id = torch.argmax(outputs[0, masked_index]).item()
            predicted_word = tokenizer.decode(predicted_id).strip()
            
            # Replace masked word
            augmented_sentence[i] = predicted_word

        generated_text = " ".join(augmented_sentence)

        
        # Calculate similarity with original text
        generated_embedding = similarity_model.encode(generated_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(original_embedding, generated_embedding).item()
        
        if similarity_score >= similarity_threshold and generated_text != text:
            with open(output_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([category, subcategory, text, generated_text])

# Example usage with a sample dataset
dataset_path = 'train.csv'
df = pd.read_csv(dataset_path)
print(df.shape)

# Apply augmentation on each row of the dataset
df.apply(augment_text, axis=1)
