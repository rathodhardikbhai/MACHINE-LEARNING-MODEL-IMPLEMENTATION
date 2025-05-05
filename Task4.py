# MACHINE LEARNING MODEL IMPLEMENTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
try:
    df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Please download the dataset first from:")
    print("https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
    raise

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df['label'].value_counts())
print("\nPercentage distribution:")
print(df['label'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title('Class Distribution')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.show()

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Missing values:")
print(df.isnull().sum())

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Shape of TF-IDF matrix for training set:", X_train_tfidf.shape)
print("Shape of TF-IDF matrix for test set:", X_test_tfidf.shape)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    
    report = classification_report(y_test, test_preds, target_names=['ham', 'spam'])
    
    cm = confusion_matrix(y_test, test_preds)
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'report': report,
        'confusion_matrix': cm
    }

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear')
}

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    result = evaluate_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    results[name] = result
    
    print(f"Training Accuracy: {result['train_accuracy']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print("\nClassification Report:")
    print(result['report'])

model_names = list(results.keys())
test_accuracies = [results[name]['test_accuracy'] for name in model_names]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=test_accuracies)
plt.title('Model Comparison - Test Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
for i, v in enumerate(test_accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

plt.figure(figsize=(15, 4))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid = {
    'tfidf__max_features': [1000, 2000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
    'clf__C': [0.1, 1, 10],  # regularization strength
    'clf__penalty': ['l2']  # regularization type
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test set accuracy with best model: {:.4f}".format(test_accuracy))

sample_emails = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!",
    "Hi John, just checking in to see if you're still coming to the meeting tomorrow at 2pm.",
    "URGENT: Your bank account has been compromised. Click this link to secure your account immediately!",
    "Hey, don't forget to bring the documents we discussed. See you at the office.",
    "You're selected for our exclusive offer! Get 80% off on your next purchase. Limited time only!"
]

predictions = best_model.predict(sample_emails)
prediction_probs = best_model.predict_proba(sample_emails)

print("\nSample Email Predictions:")
for email, pred, prob in zip(sample_emails, predictions, prediction_probs):
    print(f"\nEmail: {email}")
    print(f"Predicted: {'spam' if pred == 1 else 'ham'}")
    print(f"Probability: {prob[1]:.4f} (spam), {prob[0]:.4f} (ham)")
