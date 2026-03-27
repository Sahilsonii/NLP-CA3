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
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


class SpamDetector:
    """Base class for Spam Detection models."""
    
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.model = None
        self.training_time = None
        self.is_trained = False
        self.history = None
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
    
    def _build_lstm(self, vocab_size, embedding_dim=64, max_length=100):
        self.model = Sequential([
            Input(shape=(max_length,), name="token_ids"),
            Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding"),
            Bidirectional(LSTM(32, return_sequences=True, dropout=0.2), name="bilstm"),
            Dropout(0.2, name="dropout_1"),
            LSTM(16, dropout=0.2, name="lstm"),
            Dense(16, activation="relu", name="dense_relu"),
            Dropout(0.2, name="dropout_2"),
            Dense(1, activation="sigmoid", name="spam_probability")
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        self.model.build((None, max_length))
        self.model.summary()

    def _configure_traditional_model(self, sklearn_verbose=0):
        """Enable native sklearn/libsvm verbosity where supported."""
        if sklearn_verbose <= 0 or self.model is None:
            return

        model_params = self.model.get_params() if hasattr(self.model, "get_params") else {}

        if "verbose" in model_params:
            verbose_value = bool(sklearn_verbose) if self.model_type == "svm" else int(sklearn_verbose)
            self.model.set_params(verbose=verbose_value)
            print(f"Native sklearn verbosity enabled for {self.model_type} (level={sklearn_verbose}).")
            return

        print(
            f"Native sklearn verbosity is not available for {self.model_type}; "
            "fit will run without raw iterative logs."
        )

    def _fit_with_fallback(self, X_train, y_train):
        """Retry with single-threaded execution when multiprocessing is blocked."""
        try:
            self.model.fit(X_train, y_train)
        except PermissionError:
            if hasattr(self.model, "get_params") and "n_jobs" in self.model.get_params():
                current_n_jobs = self.model.get_params().get("n_jobs")
                if current_n_jobs != 1:
                    print("Multiprocessing is unavailable here; retrying with n_jobs=1...")
                    self.model.set_params(n_jobs=1)
                    self.model.fit(X_train, y_train)
                    return
            raise
    
    def train(self, X_train, y_train, **kwargs):
        print(f"Training {self.model_type}...")
        start_time = time.time()

        if self.model_type == "lstm":
            vocab_size = kwargs.get("vocab_size", 10000)
            max_length = kwargs.get("max_length", 100)
            epochs = kwargs.get("epochs", 10)
            batch_size = kwargs.get("batch_size", 32)
            validation_split = kwargs.get("validation_split", 0.1)
            use_class_weight = kwargs.get("use_class_weight", False)
            verbose = kwargs.get("verbose", 2)

            self._build_lstm(vocab_size, max_length=max_length)

            early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            print(f"Input shape: {X_train.shape}, Label shape: {y_train.shape}")
            print(f"Y unique values: {np.unique(y_train)}")
            print(f"Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

            fit_kwargs = {
                "epochs": epochs,
                "batch_size": batch_size,
                "validation_split": validation_split,
                "callbacks": [early_stop],
                "verbose": verbose,
            }

            if use_class_weight:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                weights = compute_class_weight("balanced", classes=classes, y=y_train)
                fit_kwargs["class_weight"] = {int(label): float(weight) for label, weight in zip(classes, weights)}
                print(f"Class weights: {fit_kwargs['class_weight']}")

            self.history = self.model.fit(
                X_train, y_train,
                **fit_kwargs,
            )
        else:
            sklearn_verbose = kwargs.get("sklearn_verbose", 0)
            self._configure_traditional_model(sklearn_verbose=sklearn_verbose)
            self._fit_with_fallback(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_trained = True
        print(f"Training completed in {self.training_time:.2f} seconds")
    
    def predict(self, X_test):
        if self.model_type == "lstm":
            predictions = self.model.predict(X_test, verbose=0).flatten()
            y_pred = (predictions > 0.5).astype(int).flatten()
            print(
                "   Debug LSTM probabilities: "
                f"min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}"
            )
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

    def get_training_diagnostics(self):
        """Return lightweight training details for console reporting."""
        if not self.is_trained:
            return {}

        if self.model_type == "logistic_regression" and hasattr(self.model, "n_iter_"):
            return {"solver_iterations": int(np.max(np.atleast_1d(self.model.n_iter_)))}

        if self.model_type == "naive_bayes":
            return {"fit_mode": "single-pass probabilistic fit"}

        if self.model_type == "random_forest" and hasattr(self.model, "estimators_"):
            return {"trees_trained": len(self.model.estimators_)}

        if self.model_type == "svm" and hasattr(self.model, "n_support_"):
            return {"support_vectors": int(np.sum(self.model.n_support_))}

        if self.model_type == "lstm" and self.history is not None:
            history = self.history.history
            diagnostics = {"epochs_ran": len(history.get("loss", []))}
            if history.get("val_accuracy"):
                diagnostics["best_val_accuracy"] = float(max(history["val_accuracy"]))
            if history.get("val_loss"):
                diagnostics["best_val_loss"] = float(min(history["val_loss"]))
            return diagnostics

        return {}


def get_model_description(model_type):
    descriptions = {
        "logistic_regression": {"name": "Logistic Regression", "description": "Linear model using logistic function"},
        "naive_bayes": {"name": "Multinomial Naive Bayes", "description": "Probabilistic classifier based on Bayes theorem"},
        "random_forest": {"name": "Random Forest", "description": "Ensemble of decision trees"},
        "svm": {"name": "Support Vector Machine", "description": "Finds optimal hyperplane to separate classes"},
        "lstm": {"name": "LSTM Neural Network", "description": "Deep learning model for sequential data"}
    }
    return descriptions.get(model_type, {"name": model_type, "description": "Unknown"})
