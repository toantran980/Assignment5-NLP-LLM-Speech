# spam_filter.py
# --------------------------------------------------------------
# Spam classifier using Bag-of-Words + MLP text classifier.
# Dataset: L06_NLP_emails.csv with columns: "text", "spam"
# --------------------------------------------------------------

import pandas as pd
import re
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK data if missing
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Clean and normalize text. Return preprocessed text."""
    if not text:
        return ""
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    
    # Tokenize text into words
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    
    # Lemmatize words
    #words = [LEMMATIZER.lemmatize(word) for word in words]

    # Stem words
    words = [STEMMER.stem(word) for word in words]
    
    # Join words back into a string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text

def main():
    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv("./data/input/L06_NLP_emails.csv")

    # Apply text preprocessing
    df["clean_text"] = df["text"].apply(preprocess)

    # -----------------------------
    # Create Bag-of-Words DTM
    # -----------------------------
    vectorizer = CountVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2)) # TODO: Construct your CBOW DTM.
    X = vectorizer.fit_transform(df["clean_text"]).toarray()
    y = df["spam"].values

    # -----------------------------
    # Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # TODO: Create your training/testing split using train_test_split.

    # -----------------------------
    # Train MLP
    # -----------------------------
    clf = MLPClassifier(hidden_layer_sizes=(300, 300, 300), max_iter=1500, random_state=42, alpha=0.001) # TODO: Use MLPClassifier to create a an MLP model for training/testing.

    clf.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nSpam Filter Accuracy: {acc * 100:.2f}%")
    print("Training complete.")


if __name__ == "__main__":
    main()
