import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Load the data
train_data = pd.read_csv('TRAIN_AUG_3_PROCESSED.csv')
test_data = pd.read_csv('TEST_PROCESSED.csv')

# Drop rows with NaN values in the relevant columns
train_data = train_data.dropna(subset=['crimeaditionalinfo', 'category']).reset_index(drop=True)
test_data = test_data.dropna(subset=['crimeaditionalinfo', 'category']).reset_index(drop=True)

# Separate features and labels
X_train = train_data['crimeaditionalinfo']
y_train = train_data['category']
X_test = test_data['crimeaditionalinfo']
y_test = test_data['category']

# Encode the labels in train set
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Filter test set to only include labels present in the train set
test_data_filtered = test_data[test_data['category'].isin(label_encoder.classes_)].reset_index(drop=True)
X_test = test_data_filtered['crimeaditionalinfo']
y_test = test_data_filtered['category']
y_test_encoded = label_encoder.transform(y_test)

# Save the label encoder for future use
joblib.dump(label_encoder, 'label_encoder.pkl')

# Apply TF-IDF Vectorization with a limited number of features
tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# Save the SVD model
joblib.dump(svd, 'svd_model.pkl')
# Adjusting parameters for better handling of class imbalance
xgb_model = XGBClassifier(
    max_depth=7,               # Increase depth to learn more complex patterns
    min_child_weight=3,
    gamma=0.1,
    learning_rate=0.05,        # Reduced learning rate for better generalization
    scale_pos_weight=10,       # Increased weight to focus on minority classes
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Initialize and train the XGBoost classifier with parameters for imbalance
# xgb_model = XGBClassifier(
#     max_depth=5,
#     min_child_weight=3,
#     gamma=0.1,
#     learning_rate=0.1,
#     scale_pos_weight=5,
#     objective='multi:softmax',
#     num_class=len(label_encoder.classes_),
#     eval_metric='mlogloss',
#     use_label_encoder=False,
#     random_state=42
# )

# Train the model
xgb_model.fit(X_train_svd, y_train_encoded)

# Save the trained model
joblib.dump(xgb_model, 'xgb_model.pkl')

# Predict on test set
y_pred = xgb_model.predict(X_test_svd)

# Get the unique classes in the filtered test set for accurate reporting
target_names = label_encoder.inverse_transform(sorted(set(y_test_encoded)))

# Calculate accuracy and macro F1 score
accuracy = accuracy_score(y_test_encoded, y_pred)
report = classification_report(y_test_encoded, y_pred, target_names=target_names, output_dict=True)

# Display the metrics
print(f"Accuracy: {accuracy}")
print(f"Macro F1 Score: {report['macro avg']['f1-score']}")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=target_names))


# # Calculate accuracy and macro F1 score
# accuracy = accuracy_score(y_test_encoded, y_pred)
# report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)

# # Display the metrics
# print(f"Accuracy: {accuracy}")
# print(f"Macro F1 Score: {report['macro avg']['f1-score']}")
# print("Classification Report:")
# print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# Output the best parameters used
print("Best Parameters:")
print(xgb_model.get_params())
