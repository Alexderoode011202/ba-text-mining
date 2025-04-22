# conda activate TM

import spacy
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

print("=================================================================================================")
print("Sentiment Analysis - VADER")
print("=================================================================================================")
# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load data
df = pd.read_csv('Project//sentiment-topic-test.tsv', sep='\t')

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Define function to convert VADER scores to category labels
def vader_to_label(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER analysis
df['vader_scores'] = df['sentence'].apply(lambda x: sid.polarity_scores(x))
df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
df['vader_sentiment'] = df['vader_compound'].apply(vader_to_label)

# Evaluate results
print("VADER Sentiment Analysis Accuracy:", accuracy_score(df['sentiment'], df['vader_sentiment']))
print("\nClassification Report:")
print(classification_report(df['sentiment'], df['vader_sentiment']))

# Error analysis
errors = df[df['sentiment'] != df['vader_sentiment']]
print(f"\nNumber of samples misclassified by VADER: {len(errors)}")
print(errors[['sentence', 'sentiment', 'vader_sentiment', 'vader_compound']])

# Save VADER results
df.to_csv('vader_results.csv', index=False)
# Load NER-test.tsv data
ner_data = pd.read_csv('/home/alex/Documents/GitHub/ba-text-mining/Project/NER-test.tsv', sep='\t')

# Build sentence dictionary, grouped by sentence_id
sentences = {}
for sentence_id, group in ner_data.groupby('sentence_id'):
    sentences[sentence_id] = {
        'tokens': group['token'].tolist(),
        'true_tags': group['BIO_NER_tag'].tolist()
    }

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Entity type mapping function
def map_entity_type(spacy_type):
    mapping = {
        'PERSON': 'PERSON',
        'ORG': 'ORG',
        'GPE': 'LOC',
        'LOC': 'LOC',
        'WORK_OF_ART': 'WORK_OF_ART',
        'FAC': 'LOC'
    }
    return mapping.get(spacy_type, 'MISC')

# Generate predictions for each sentence
for sentence_id, sentence_data in sentences.items():
    # Reconstruct original sentence text
    text = ' '.join(sentence_data['tokens'])
    
    # Predict using spaCy
    doc = nlp(text)
    
    # Initialize prediction labels (all "O")
    pred_tags = ['O'] * len(sentence_data['tokens'])
    
    # Fill prediction labels based on spaCy identified entities
    token_index = 0
    for token in doc:
        if token.ent_type_:
            # Map spaCy entity types to our tag format
            entity_type = map_entity_type(token.ent_type_)
            if token.ent_iob_ == 'B':
                pred_tags[token_index] = f'B-{entity_type}'
            elif token.ent_iob_ == 'I':
                pred_tags[token_index] = f'I-{entity_type}'
        token_index += 1
    
    # Save predictions
    sentence_data['pred_tags'] = pred_tags

# Evaluate NERC performance
true_tags = []
pred_tags = []
for sentence_data in sentences.values():
    true_tags.extend(sentence_data['true_tags'])
    pred_tags.extend(sentence_data['pred_tags'])

print(classification_report(true_tags, pred_tags))
print("=================================================================================================")
print("Sentiment Analysis - VADER")
print("=================================================================================================")
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

# Ensure VADER lexicon is downloaded
print("Ensuring VADER lexicon is downloaded...")
nltk.download('vader_lexicon')

# Load data
print("Loading data...")
df = pd.read_csv('Project//sentiment-topic-test.tsv', sep='\t')

# Initialize VADER
print("Initializing VADER...")
sid = SentimentIntensityAnalyzer()

# Define function to convert VADER scores to category labels
def vader_to_label(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER analysis
print("Applying VADER analysis...")
df['vader_scores'] = df['sentence'].apply(lambda x: sid.polarity_scores(x))
df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
df['vader_sentiment'] = df['vader_compound'].apply(vader_to_label)

# Evaluate results
print("Evaluating results...")
print("VADER Sentiment Analysis Accuracy:", accuracy_score(df['sentiment'], df['vader_sentiment']))
print("\nClassification Report:")
print(classification_report(df['sentiment'], df['vader_sentiment']))

# Error analysis
print("Performing error analysis...")
errors = df[df['sentiment'] != df['vader_sentiment']]
print(f"\nNumber of samples misclassified by VADER: {len(errors)}")
print(errors[['sentence', 'sentiment', 'vader_sentiment', 'vader_compound']])

# Save VADER results
df.to_csv('vader_results.csv', index=False)

print("=================================================================================================")
print("Sentiment Analysis - Machine Learning")
print("=================================================================================================")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load data
print("Loading data...")
vader_df = pd.read_csv('vader_results.csv')

# Split data into training and test sets
# Note: In real applications, you might need more data to train
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    vader_df['sentence'], vader_df['sentiment'], test_size=0.3, random_state=42
)

# Create processing pipeline: TF-IDF + SVM
print("Creating processing pipeline: TF-IDF + SVM...")
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])

# Train model
print("Training model...")
sentiment_pipeline.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = sentiment_pipeline.predict(X_test)

# Evaluate results
print("Evaluating results...")
print("SVM Sentiment Analysis Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Error analysis
print("Performing error analysis...")
test_df = pd.DataFrame({
    'sentence': X_test,
    'true_sentiment': y_test,
    'pred_sentiment': y_pred
})
errors = test_df[test_df['true_sentiment'] != test_df['pred_sentiment']]
print(f"\nNumber of samples misclassified by SVM: {len(errors)}")
print(errors)

# Predict on all data (for comparison with VADER)
print("Predicting on all data (for comparison with VADER)...")
vader_df['ml_sentiment'] = sentiment_pipeline.predict(vader_df['sentence'])
vader_df.to_csv('sentiment_results.csv', index=False)


# =================================================================================================
print("=================================================================================================")
print("NER Classification - SVM")
print("=================================================================================================")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Prepare features for each token
def extract_token_features(sentence_tokens, index):
    token = sentence_tokens[index]
    features = {
        'bias': 1.0,
        'token.lower()': token.lower(),
        'token[-3:]': token[-3:],
        'token[-2:]': token[-2:],
        'token.isupper()': token.isupper(),
        'token.istitle()': token.istitle(),
        'token.isdigit()': token.isdigit(),
    }
    if index > 0:
        token1 = sentence_tokens[index - 1]
        features.update({
            '-1:token.lower()': token1.lower(),
            '-1:token.istitle()': token1.istitle(),
            '-1:token.isupper()': token1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if index < len(sentence_tokens) - 1:
        token1 = sentence_tokens[index + 1]
        features.update({
            '+1:token.lower()': token1.lower(),
            '+1:token.istitle()': token1.istitle(),
            '+1:token.isupper()': token1.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

# Build feature and label sets
X, y = [], []
for sentence_data in sentences.values():
    tokens = sentence_data['tokens']
    tags = sentence_data['true_tags']
    for i in range(len(tokens)):
        X.append(extract_token_features(tokens, i))
        y.append(tags[i])

# Vectorize features
print("Vectorizing features...")
vec = DictVectorizer(sparse=True)
X_vectorized = vec.fit_transform(X)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
print("Training SVM model for NER...")
svm_ner_model = LinearSVC()
svm_ner_model.fit(X_train, y_train)

# Predict and evaluate
print("Evaluating model...")
y_pred = svm_ner_model.predict(X_test)

print("NER Classification Report (SVM):")
print(classification_report(y_test, y_pred))

# =================================================================================================
print("=================================================================================================")
print("Topic Analysis - SVM")
print("=================================================================================================")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load data
print("Loading data...")
df = pd.read_csv('Project//sentiment-topic-test.tsv', sep='\t')

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['sentence'], df['topic'], test_size=0.3, random_state=42
)

# Create processing pipeline: TF-IDF + SVM
print("Creating processing pipeline: TF-IDF + SVM...")
topic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LinearSVC())
])

# Train model
print("Training model...")
topic_pipeline.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = topic_pipeline.predict(X_test)

# Evaluate results
print("Evaluating results...")
print("Topic Classification Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict on all data
print("Predicting on all data...")
df['predicted_topic'] = topic_pipeline.predict(df['sentence'])

# ================================================================================================= 
# Combine the results of both methods
print("Combining the results of both methods...")
results_df = pd.read_csv('sentiment_results.csv')
comparison_df = results_df[['sentence', 'sentiment', 'vader_sentiment', 'ml_sentiment']]

# Find examples where both methods are wrong
print("Finding examples where both methods are wrong...")
both_wrong = comparison_df[(comparison_df['sentiment'] != comparison_df['vader_sentiment']) & 
                           (comparison_df['sentiment'] != comparison_df['ml_sentiment'])]

# Find examples where VADER is correct but ML is wrong
print("Finding examples where VADER is correct but ML is wrong...")
vader_right_ml_wrong = comparison_df[(comparison_df['sentiment'] == comparison_df['vader_sentiment']) & 
                                     (comparison_df['sentiment'] != comparison_df['ml_sentiment'])]

# Find examples where ML is correct but VADER is wrong
print("Finding examples where ML is correct but VADER is wrong...")
ml_right_vader_wrong = comparison_df[(comparison_df['sentiment'] != comparison_df['vader_sentiment']) & 
                                     (comparison_df['sentiment'] == comparison_df['ml_sentiment'])]

print(f"Number of samples where both methods are wrong: {len(both_wrong)}")
print(f"Number of samples where VADER is correct but ML is wrong: {len(vader_right_ml_wrong)}")
print(f"Number of samples where ML is correct but VADER is wrong: {len(ml_right_vader_wrong)}")
