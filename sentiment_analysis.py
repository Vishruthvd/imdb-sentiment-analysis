import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load real IMDb dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)   # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

# Apply cleaning
df["clean_review"] = df["review"].apply(clean_text)

# Features and labels
X = df["clean_review"]
y = df["sentiment"]

# Convert text to numerical form
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom input test
sample_review = ["The movie was absolutely fantastic and inspiring"]
sample_clean = [clean_text(sample_review[0])]
sample_vec = vectorizer.transform(sample_clean)

print("Sample Prediction:", model.predict(sample_vec)[0])
