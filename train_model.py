# train_model.py
# DeepFlixAI - CNN Multi-Label Genre Training

import os
import sys
import json
import pickle
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pymysql
import pymysql.cursors
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ==========================================
# BASE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "ott_recommendation_db1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 15,
    "read_timeout": 60,
    "write_timeout": 60
}

# SSL support - same pattern as app.py
# Enable SSL only when explicitly configured
DB_SSL_CA = os.getenv("DB_SSL_CA", "").strip()

if DB_SSL_CA and os.path.isfile(DB_SSL_CA):
    DB_CONFIG["ssl"] = {"ca": DB_SSL_CA}


# ==========================================
# MODEL PARAMETERS
# ==========================================

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 64
FILTERS = 64
KERNEL_SIZE = 5
DROPOUT_RATE = 0.4

EPOCHS = 10
BATCH_SIZE = 8


# ==========================================
# DATABASE
# ==========================================

def get_connection():

    print("Connecting to database...", flush=True)

    try:
        conn = pymysql.connect(**DB_CONFIG)

        print("Database connection successful.", flush=True)

        return conn

    except Exception as e:

        print(f"Database connection failed: {e}", flush=True)

        raise


# ==========================================
# MODEL
# ==========================================

def create_classification_model(
    vocab_size,
    max_sequence_length,
    embedding_dim,
    filters,
    kernel_size,
    dropout_rate,
    num_classes
):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),

        Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu"
        ),

        GlobalMaxPooling1D(),

        Dense(64, activation="relu"),

        Dropout(dropout_rate),

        Dense(
            32,
            activation="relu",
            name="embedding_layer"
        ),

        Dense(
            num_classes,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return model


# ==========================================
# TRAINING & VALIDATION GRAPH
# ==========================================

def plot_history(history, filename):

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename, dpi=120)

    plt.close()

    print(f"Training graph saved: {filename}", flush=True)


# ==========================================
# CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(y_true, y_pred, filename):

    # Multi-label evaluation converted into a binary
    # present/absent confusion matrix across all genres.

    true_flat = y_true.flatten()

    pred_flat = y_pred.flatten()

    cm = confusion_matrix(
        true_flat,
        pred_flat,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Absent", "Present"],
        yticklabels=["Absent", "Present"]
    )

    plt.title("Genre Prediction Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(filename, dpi=120)

    plt.close()

    print(f"Confusion matrix saved: {filename}", flush=True)

    return cm


# ==========================================
# MAIN TRAINING
# ==========================================

def train_and_save_model():

    print("=" * 50, flush=True)

    print("DEEPFLIX AI TRAINING STARTED", flush=True)

    print("=" * 50, flush=True)

    conn = get_connection()

    try:

        # --------------------------------------
        # 1. LOAD DATA
        # --------------------------------------

        with conn.cursor() as cursor:

            cursor.execute(
                "SELECT id, story, genre FROM contents"
            )

            contents = cursor.fetchall()

        if not contents:

            raise ValueError(
                "No content records found in contents table."
            )

        df = pd.DataFrame(contents)

        if "story" not in df.columns or "genre" not in df.columns:

            raise ValueError(
                "Database must contain story and genre columns."
            )

        df["story"] = df["story"].fillna("").astype(str)

        df["genre"] = df["genre"].fillna("").astype(str)

        # --------------------------------------
        # 2. CLEAN DATA
        # --------------------------------------

        df["genre"] = df["genre"].apply(
            lambda x: [
                g.strip()
                for g in x.split(",")
                if g.strip()
            ]
        )

        df = df[
            (df["story"].str.strip() != "") &
            (df["genre"].apply(len) > 0)
        ].reset_index(drop=True)

        if len(df) < 2:

            raise ValueError(
                "At least 2 valid movies with story and genre are required."
            )

        print(
            f"Valid training records: {len(df)}",
            flush=True
        )

        # --------------------------------------
        # 3. LABEL PROCESSING
        # --------------------------------------

        mlb = MultiLabelBinarizer()

        y_labels = mlb.fit_transform(df["genre"])

        class_labels = list(mlb.classes_)

        num_classes = len(class_labels)

        if num_classes < 1:

            raise ValueError(
                "No genre labels available."
            )

        print(
            f"Genres detected: {class_labels}",
            flush=True
        )

        # --------------------------------------
        # 4. TOKENIZATION
        # --------------------------------------

        tokenizer = Tokenizer(
            num_words=VOCAB_SIZE,
            oov_token="<unk>"
        )

        tokenizer.fit_on_texts(df["story"])

        sequences = tokenizer.texts_to_sequences(
            df["story"]
        )

        padded_sequences = pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="pre",
            truncating="pre"
        )

        X = np.asarray(
            padded_sequences,
            dtype=np.int32
        )

        y = np.asarray(
            y_labels,
            dtype=np.float32
        )

        # --------------------------------------
        # 5. TRAIN / VALIDATION SPLIT
        # --------------------------------------

        validation_count = max(
            1,
            int(np.ceil(len(X) * 0.2))
        )

        validation_count = min(
            validation_count,
            len(X) - 1
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_count,
            random_state=42
        )

        print(
            f"Training samples: {len(X_train)}",
            flush=True
        )

        print(
            f"Validation samples: {len(X_val)}",
            flush=True
        )

        # --------------------------------------
        # 6. BUILD MODEL
        # --------------------------------------

        tf.keras.backend.clear_session()

        model = create_classification_model(
            VOCAB_SIZE,
            MAX_SEQUENCE_LENGTH,
            EMBEDDING_DIM,
            FILTERS,
            KERNEL_SIZE,
            DROPOUT_RATE,
            num_classes
        )

        model.summary()

        # --------------------------------------
        # 7. TRAIN
        # --------------------------------------

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

        ]

        history = model.fit(

            X_train,

            y_train,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            verbose=1,

            validation_data=(X_val, y_val),

            callbacks=callbacks

        )

        print(
            "Training completed.",
            flush=True
        )

        # --------------------------------------
        # 8. PREDICTION
        # --------------------------------------

        y_pred_probs = model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        # Threshold for multi-label predictions
        y_pred = (y_pred_probs >= 0.5).astype(int)

        # --------------------------------------
        # 9. SAVE MODEL
        # --------------------------------------

        model_path = os.path.join(
            DATA_DIR,
            "pretrained_model.keras"
        )

        temp_model_path = os.path.join(
            DATA_DIR,
            "pretrained_model_temp.keras"
        )

        model.save(temp_model_path)

        os.replace(
            temp_model_path,
            model_path
        )

        print(
            f"Model saved: {model_path}",
            flush=True
        )

        # --------------------------------------
        # 10. SAVE TOKENIZER
        # --------------------------------------

        tokenizer_path = os.path.join(
            DATA_DIR,
            "tokenizer.pkl"
        )

        temp_tokenizer_path = tokenizer_path + ".tmp"

        with open(
            temp_tokenizer_path,
            "wb"
        ) as handle:

            pickle.dump(
                tokenizer,
                handle
            )

        os.replace(
            temp_tokenizer_path,
            tokenizer_path
        )

        # --------------------------------------
        # 11. SAVE MLB
        # --------------------------------------

        mlb_path = os.path.join(
            DATA_DIR,
            "mlb.pkl"
        )

        temp_mlb_path = mlb_path + ".tmp"

        with open(
            temp_mlb_path,
            "wb"
        ) as handle:

            pickle.dump(
                mlb,
                handle
            )

        os.replace(
            temp_mlb_path,
            mlb_path
        )

        # --------------------------------------
        # 12. SAVE CONFUSION MATRIX
        # --------------------------------------

        cm_path = os.path.join(
            IMG_DIR,
            "confusion_matrix.png"
        )

        cm = save_confusion_matrix(
            y_val.astype(int),
            y_pred,
            cm_path
        )

        # --------------------------------------
        # 13. SAVE TRAINING GRAPH
        # --------------------------------------

        history_path = os.path.join(
            IMG_DIR,
            "accuracy_loss.png"
        )

        plot_history(
            history,
            history_path
        )

        # --------------------------------------
        # 14. SAVE METRICS
        # --------------------------------------

        metrics = {

            "confusion_matrix": cm.tolist(),

            "labels": [
                "Absent",
                "Present"
            ],

            "genres": class_labels,

            "accuracy": history.history.get(
                "binary_accuracy",
                []
            ),

            "val_accuracy": history.history.get(
                "val_binary_accuracy",
                []
            ),

            "loss": history.history.get(
                "loss",
                []
            ),

            "val_loss": history.history.get(
                "val_loss",
                []
            ),

            "total_samples": len(df),

            "training_samples": len(X_train),

            "validation_samples": len(X_val),

            "epochs_completed": len(
                history.history.get("loss", [])
            )

        }

        metrics_path = os.path.join(
            DATA_DIR,
            "model_metrics.json"
        )

        temp_metrics_path = metrics_path + ".tmp"

        with open(
            temp_metrics_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        os.replace(
            temp_metrics_path,
            metrics_path
        )

        # --------------------------------------
        # 15. SUCCESS
        # --------------------------------------

        print("=" * 50, flush=True)

        print(
            "MODEL TRAINING SUCCESSFULLY COMPLETED",
            flush=True
        )

        print("=" * 50, flush=True)

        return True

    except Exception as e:

        print(
            f"TRAINING ERROR: {e}",
            flush=True
        )

        traceback.print_exc()

        raise

    finally:

        if conn:

            conn.close()

        tf.keras.backend.clear_session()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    try:

        train_and_save_model()

        sys.exit(0)

    except Exception:

        print(
            "Training process failed.",
            flush=True
        )

        sys.exit(1)