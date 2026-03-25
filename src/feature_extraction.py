"""
Feature Extraction Module for SMS Spam Detection

This module provides classes for extracting features from text:
- TF-IDF Vectorization for traditional ML models
- Sequence tokenization for deep learning models
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TFIDFExtractor:
    """
    TF-IDF Feature Extractor for traditional ML models.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text 
    into numerical features by considering word importance.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF Extractor.
        
        Parameters:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to extract
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """
        Fit the vectorizer on training texts.
        
        Parameters:
            texts (list): List of text strings
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"TF-IDF Vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features.
        
        Parameters:
            texts (list): List of text strings
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit and transform texts in one step.
        
        Parameters:
            texts (list): List of text strings
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        self.is_fitted = True
        features = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF Vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
        return features
    
    def get_feature_names(self):
        """Get list of feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save vectorizer to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")
    
    def load(self, filepath):
        """Load vectorizer from file."""
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")


class SequenceExtractor:
    """
    Sequence Feature Extractor for deep learning models.
    
    Converts text into sequences of integers for LSTM/RNN models.
    """
    
    def __init__(self, max_words=10000, max_length=100):
        """
        Initialize Sequence Extractor.
        
        Parameters:
            max_words (int): Maximum vocabulary size
            max_length (int): Maximum sequence length (padding)
        """
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.is_fitted = False
    
    def fit(self, texts):
        """
        Fit the tokenizer on training texts.
        
        Parameters:
            texts (list): List of text strings
        """
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        vocab_size = min(len(self.tokenizer.word_index), self.max_words)
        print(f"Tokenizer fitted with vocabulary size: {vocab_size}")
    
    def transform(self, texts):
        """
        Transform texts to padded sequences.
        
        Parameters:
            texts (list): List of text strings
            
        Returns:
            np.array: Padded sequence matrix
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding="post")
        return padded
    
    def fit_transform(self, texts):
        """
        Fit and transform texts in one step.
        
        Parameters:
            texts (list): List of text strings
            
        Returns:
            np.array: Padded sequence matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_vocabulary_size(self):
        """Get actual vocabulary size."""
        return min(len(self.tokenizer.word_index) + 1, self.max_words)
    
    def save(self, filepath):
        """Save tokenizer to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """Load tokenizer from file."""
        with open(filepath, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.is_fitted = True
        print(f"Tokenizer loaded from {filepath}")


def extract_manual_features(texts):
    """
    Extract manual features from texts.
    
    These features can be used alongside TF-IDF for better performance.
    
    Parameters:
        texts (list): List of text strings
        
    Returns:
        np.array: Feature matrix with manual features
    """
    features = []
    
    for text in texts:
        text_features = {
            "length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
            "capital_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "special_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
        }
        features.append(list(text_features.values()))
    
    return np.array(features)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "free prize winner call now",
        "hey how are you doing today",
        "urgent claim your cash prize",
        "meeting tomorrow at office"
    ]
    
    print("=" * 50)
    print("TF-IDF EXTRACTION EXAMPLE")
    print("=" * 50)
    
    tfidf = TFIDFExtractor(max_features=100)
    tfidf_features = tfidf.fit_transform(sample_texts)
    print(f"TF-IDF shape: {tfidf_features.shape}")
    
    print("\n" + "=" * 50)
    print("SEQUENCE EXTRACTION EXAMPLE")
    print("=" * 50)
    
    seq = SequenceExtractor(max_words=100, max_length=10)
    seq_features = seq.fit_transform(sample_texts)
    print(f"Sequence shape: {seq_features.shape}")
    print(f"Sample sequence: {seq_features[0]}")
