# train_model.py
# This script is called by the Flask app to train the NLP model.
# It reads data from the database, trains a CNN model, and saves it.

import pandas as pd
import pymysql.cursors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'ott_recommendation_db1',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# --- Model Parameters ---
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 128
FILTERS = 128
KERNEL_SIZE = 5
DROPOUT_RATE = 0.5
EPOCHS = 20
BATCH_SIZE = 32

def get_connection():
    """Establishes a new database connection."""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def create_classification_model(vocab_size, max_sequence_length, embedding_dim, filters, kernel_size, dropout_rate, num_classes):
    """Creates a 1D CNN classification model for text representation."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu', name='embedding_layer'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_history(history, filename):
    """Plots training and validation loss and accuracy and saves the plot."""
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_confusion_matrix(y_true, y_pred, filename, labels):
    """Plots a confusion matrix and saves the plot."""
    y_true_labels = [str(label) for label in y_true]
    y_pred_labels = [str(label) for label in y_pred]

    labels = sorted(list(set(y_true_labels) | set(y_pred_labels)))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

    return cm, labels  # 🔥 IMPORTANT: return cm for JSON saving

def train_and_save_model():
    """Fetches data, trains the model, and saves the model and tokenizer."""
    print("Starting model training process...")

    conn = get_connection()
    if not conn:
        print("Model training aborted due to database connection failure.")
        return

    try:
        # --- 1. Fetch data from the database ---
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, story, genre FROM contents")
            contents = cursor.fetchall()

        if not contents:
            print("No content found in the database. Cannot train model.")
            return

        df = pd.DataFrame(contents)

        df['story'] = df['story'].fillna('')
        df['genre'] = df['genre'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
        
        df = df[df['genre'].apply(len) > 0].reset_index(drop=True)

        if df.empty:
            print("No valid content with genres found. Cannot train model.")
            return
            
        print(f"Fetched and processed {len(df)} content items from the database.")

        # --- 2. Preprocess ---
        mlb = MultiLabelBinarizer()
        y_labels = mlb.fit_transform(df['genre'])
        
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
        tokenizer.fit_on_texts(df['story'])

        sequences = tokenizer.texts_to_sequences(df['story'])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
        class_labels = mlb.classes_

        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, y_labels, test_size=0.2, random_state=42
        )

        # --- 3. Model ---
        print("Creating and training CNN classification model...")
        model = create_classification_model(
            VOCAB_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
            FILTERS, KERNEL_SIZE, DROPOUT_RATE, len(class_labels)
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=(X_val, y_val)
        )

        print("Model training complete.")

        # --- 4. Save ---
        if not os.path.exists('data'):
            os.makedirs('data')
        
        model.save('data/pretrained_model.keras')

        with open('data/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle)

        with open('data/mlb.pkl', 'wb') as handle:
            pickle.dump(mlb, handle)

        # --- 5. Predictions ---
        y_pred_probs = model.predict(X_val)
        y_pred_labels_indices = np.argmax(y_pred_probs, axis=1)
        y_true_labels_indices = np.argmax(y_val, axis=1)
        
        y_pred_labels = [class_labels[i] for i in y_pred_labels_indices]
        y_true_labels = [class_labels[i] for i in y_true_labels_indices]

        img_dir = 'static/img'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        cm, cm_labels = plot_confusion_matrix(
            y_true_labels,
            y_pred_labels,
            os.path.join(img_dir, 'confusion_matrix.png'),
            list(class_labels)
        )

        # ===============================
        # 🔥 SAVE METRICS JSON (FINAL)
        # ===============================
        metrics = {
            "confusion_matrix": cm.tolist(),
            "labels": list(cm_labels),
            "accuracy": history.history['accuracy'],
            "val_accuracy": history.history['val_accuracy'],
            "loss": history.history['loss'],
            "val_loss": history.history['val_loss']
        }

        with open("data/model_metrics.json", "w") as f:
            json.dump(metrics, f)

        print("✅ Metrics JSON saved successfully")
        print("Training process finished.")

    except Exception as e:
        print(f"An error occurred during training: {e}")

    finally:
        conn.close()


if __name__ == '__main__':
    train_and_save_model()