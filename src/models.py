"""
Models Module for SMS Spam Detection

This module provides implementations of 5 different models:
1. Logistic Regression
2. Naive Bayes (MultinomialNB)
3. Random Forest
4. Support Vector Machine (SVM)
5. LSTM Neural Network
"""

import numpy as np
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


class SpamDetector:
    """Base class for Spam Detection models."""
    
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.model = None
        self.training_time = None
        self.is_trained = False
        self._initialize_model()
    
    def _initialize_model(self):
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        elif self.model_type == "naive_bayes":
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == "svm":
            self.model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
        elif self.model_type == "lstm":
            self.model = None
    
    def _build_lstm(self, vocab_size, embedding_dim=128, max_length=100):
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.summary()
    
    def train(self, X_train, y_train, **kwargs):
        print(f"Training {self.model_type}...")
        start_time = time.time()

        if self.model_type == "lstm":
            vocab_size = kwargs.get("vocab_size", 10000)
            max_length = kwargs.get("max_length", 100)
            epochs = kwargs.get("epochs", 10)
            batch_size = kwargs.get("batch_size", 32)

            self._build_lstm(vocab_size, max_length=max_length)

            # Calculate class weights for imbalanced dataset
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {i: w for i, w in zip(classes, weights)}

            early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            print(f"Input shape: {X_train.shape}, Label shape: {y_train.shape}")
            print(f"Y unique values: {np.unique(y_train)}")
            print(f"Class weights: {class_weight}")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                class_weight=class_weight,
                callbacks=[early_stop],
                verbose=1
            )
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"Training completed in {self.training_time:.2f} seconds")
    
    def predict(self, X_test):
        if self.model_type == "lstm":
            predictions = self.model.predict(X_test, verbose=0)
            y_pred = (predictions > 0.5).astype(int).flatten()
            # Debug info
            unique_preds = np.unique(predictions)
            print(f"   Debug LSTM predictions: min={unique_preds.min():.4f}, max={unique_preds.max():.4f}, mean={unique_preds.mean():.4f}")
            print(f"   Predictions distribution: 0={np.sum(y_pred==0)}, 1={np.sum(y_pred==1)}")
            return y_pred
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if self.model_type == "lstm":
            proba = self.model.predict(X_test, verbose=0).flatten()
            return np.column_stack([1 - proba, proba])
        return self.model.predict_proba(X_test)
    
    def save(self, filepath):
        if self.model_type == "lstm":
            self.model.save(filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)

    def load(self, filepath):
        if self.model_type == "lstm":
            self.model = tf.keras.models.load_model(filepath)
        else:
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
        self.is_trained = True


def get_model_description(model_type):
    descriptions = {
        "logistic_regression": {"name": "Logistic Regression", "description": "Linear model using logistic function"},
        "naive_bayes": {"name": "Multinomial Naive Bayes", "description": "Probabilistic classifier based on Bayes theorem"},
        "random_forest": {"name": "Random Forest", "description": "Ensemble of decision trees"},
        "svm": {"name": "Support Vector Machine", "description": "Finds optimal hyperplane to separate classes"},
        "lstm": {"name": "LSTM Neural Network", "description": "Deep learning model for sequential data"}
    }
    return descriptions.get(model_type, {"name": model_type, "description": "Unknown"})
