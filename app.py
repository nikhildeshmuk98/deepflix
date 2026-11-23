"""DeepFlix - Flask OTT app with hybrid recommender (fixed & hardened)."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Must be set BEFORE tensorflow is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import io
import json
import pickle
import re
import subprocess
import sys
import threading
import traceback
import time
import uuid

from datetime import datetime
from functools import wraps
from urllib.parse import parse_qs, urlparse

import numpy as np
import pymysql
import pymysql.cursors

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from rapidfuzz import fuzz
from textblob import TextBlob
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.secret_key = os.getenv('SECRET_KEY', 'super_secret_hybrid_key')

# Google Identity Services (set GOOGLE_CLIENT_ID in your environment/.env)
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '').strip()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_FILE = os.path.join(DATA_DIR, 'pretrained_model.keras')
TOKENIZER_FILE = os.path.join(DATA_DIR, 'tokenizer.pkl')
METRICS_FILE = os.path.join(DATA_DIR, 'model_metrics.json')
MAX_SEQUENCE_LENGTH = 500

# Set ENABLE_TEXT_MODEL=0 on Render to skip TensorFlow completely (saves ~400MB RAM)
ENABLE_TEXT_MODEL = os.getenv('ENABLE_TEXT_MODEL', '1') == '1'

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'db': os.getenv('DB_NAME', 'ott_recommendation_db1'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10,
}

DB_SSL_CA = os.getenv('DB_SSL_CA', os.path.join(BASE_DIR, 'certs', 'ca.pem'))
if os.path.isfile(DB_SSL_CA):
    DB_CONFIG['ssl'] = {'ca': DB_SSL_CA}


# =========================================================
# DATABASE HELPERS
# =========================================================
def get_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None


def safe_fetchall(cursor, sql, params=()):
    """Run a query that may fail on older schemas; return [] instead of crashing."""
    try:
        cursor.execute(sql, params)
        return cursor.fetchall()
    except Exception as e:
        print("Optional query failed:", e)
        return []


def safe_delete(cursor, table, column, value):
    """Delete rows from a dependent table; ignore tables that don't exist.
    `table` and `column` are always hard-coded constants (never user input)."""
    try:
        cursor.execute(f"DELETE FROM {table} WHERE {column}=%s", (value,))
    except Exception as e:
        print(f"safe_delete skipped {table}.{column}: {e}")


def ensure_unique(cursor, table, index_name, columns):
    try:
        cursor.execute(f"SHOW INDEX FROM {table} WHERE Key_name=%s", (index_name,))
        if not cursor.fetchone():
            cursor.execute(
                f"ALTER TABLE {table} ADD UNIQUE KEY {index_name} ({', '.join(columns)})"
            )
    except Exception as e:
        print(f"Unique key {table}.{index_name} not added: {e}")


def setup_database():
    """Creates the extra tables needed for the hybrid system if missing."""
    conn = get_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cursor:
            for column, ddl in (
                ('age', "ALTER TABLE users ADD COLUMN age INT DEFAULT 25"),
                ('gender', "ALTER TABLE users ADD COLUMN gender VARCHAR(20) DEFAULT 'Other'"),
            ):
                cursor.execute("SHOW COLUMNS FROM users LIKE %s", (column,))
                if not cursor.fetchone():
                    cursor.execute(ddl)

            # Idempotent Google sign-in columns; existing user data is preserved.
            for column, ddl in (
                ('email', "ALTER TABLE users ADD COLUMN email VARCHAR(255) NULL"),
                ('google_id', "ALTER TABLE users ADD COLUMN google_id VARCHAR(255) NULL"),
                ('auth_provider', "ALTER TABLE users ADD COLUMN auth_provider VARCHAR(30) NOT NULL DEFAULT 'local'"),
            ):
                cursor.execute("SHOW COLUMNS FROM users LIKE %s", (column,))
                if not cursor.fetchone():
                    cursor.execute(ddl)

            # Best-effort indexes; duplicate legacy values must not stop app startup.
            try:
                cursor.execute("SHOW INDEX FROM users WHERE Key_name='uq_users_google_id'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE users ADD UNIQUE KEY uq_users_google_id (google_id)")
            except Exception as index_error:
                print('Google ID index not added:', index_error)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    rating FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (content_id) REFERENCES contents(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    review_text TEXT,
                    sentiment_score FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (content_id) REFERENCES contents(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS social_graph (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    follower_id INT,
                    followed_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (follower_id) REFERENCES users(id),
                    FOREIGN KEY (followed_id) REFERENCES users(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS likes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_hearts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_heart (user_id, content_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_status (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    trained_at DATETIME,
                    accuracy FLOAT NULL
                )
            """)
            ensure_unique(cursor, 'ratings', 'uniq_user_content', ('user_id', 'content_id'))
        conn.commit()
        print("✅ Database schema ready.")
    except Exception as e:
        print(f"❌ DB Setup Error: {e}")
    finally:
        conn.close()


# =========================================================
# NLP MODEL (lazy, loaded in a background thread)
# =========================================================
MODEL = None
TOKENIZER = None
EMBEDDING_MODEL = None
_NLP_LOCK = threading.Lock()
_NLP_FAILED = False
_EMB_LOCK = threading.Lock()
_EMB_CACHE = {'key': None, 'ids': None, 'matrix': None}


# ==========================================
# NLP + CNN MODEL LOADING
# ==========================================

def text_model_ready():
    """
    Check whether the main model, tokenizer,
    and CNN embedding model are all loaded.
    """
    return (
        MODEL is not None
        and TOKENIZER is not None
        and EMBEDDING_MODEL is not None
    )


def load_nlp_model(force=False):
    """
    Load the trained TensorFlow CNN model,
    tokenizer, and embedding extraction model.

    Thread-safe model loading.
    """

    global MODEL, TOKENIZER, EMBEDDING_MODEL, _NLP_FAILED

    # --------------------------------------
    # TEXT MODEL ENABLE CHECK
    # --------------------------------------

    if not ENABLE_TEXT_MODEL:
        print("⚠️ Text model is disabled.")
        return False

    # --------------------------------------
    # LOCK: PREVENT MULTIPLE MODEL LOADS
    # --------------------------------------

    with _NLP_LOCK:

        # ----------------------------------
        # FORCE RELOAD
        # ----------------------------------

        if force:

            MODEL = None
            TOKENIZER = None
            EMBEDDING_MODEL = None
            _NLP_FAILED = False

            with _EMB_LOCK:
                _EMB_CACHE.update(
                    key=None,
                    ids=None,
                    matrix=None
                )

            print("🔄 NLP model reload requested.")

        # ----------------------------------
        # ALREADY LOADED
        # ----------------------------------

        if text_model_ready():
            print("✅ NLP model already loaded.")
            return True

        # ----------------------------------
        # PREVIOUS LOAD FAILURE
        # ----------------------------------

        if _NLP_FAILED:
            print("⚠️ Previous NLP model loading failed.")
            return False

        # ----------------------------------
        # CHECK MODEL FILES
        # ----------------------------------

        if not os.path.isfile(MODEL_FILE):
            print(f"❌ Model file missing: {MODEL_FILE}")
            _NLP_FAILED = True
            return False

        if not os.path.isfile(TOKENIZER_FILE):
            print(f"❌ Tokenizer file missing: {TOKENIZER_FILE}")
            _NLP_FAILED = True
            return False

        # ----------------------------------
        # LOAD MODEL + TOKENIZER
        # ----------------------------------

        try:

            import tensorflow as tf

            from tensorflow.keras.models import load_model

            print("🔄 Loading CNN model...")

            # Load trained model without compilation.
            # This avoids unnecessary optimizer loading.
            model = load_model(
                MODEL_FILE,
                compile=False
            )

            print("✅ CNN model loaded.")

            # ----------------------------------
            # LOAD TOKENIZER
            # ----------------------------------

            print("🔄 Loading tokenizer...")

            with open(TOKENIZER_FILE, "rb") as handle:

                tokenizer = pickle.load(handle)

            print("✅ Tokenizer loaded.")

            # ----------------------------------
            # CHECK EMBEDDING LAYER
            # ----------------------------------

            try:

                embedding_layer = model.get_layer(
                    "embedding_layer"
                )

            except ValueError:

                raise ValueError(
                    "The trained model does not contain "
                    "a layer named 'embedding_layer'. "
                    "Check train_model.py architecture."
                )

            # ----------------------------------
            # BUILD EMBEDDING EXTRACTION MODEL
            # ----------------------------------

            print("🔄 Creating CNN embedding model...")

            embedding_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=embedding_layer.output
            )

            # ----------------------------------
            # VERIFY MODEL INPUT
            # ----------------------------------

            if model.inputs is None:
                raise ValueError(
                    "CNN model input is missing."
                )

            if embedding_model is None:
                raise ValueError(
                    "CNN embedding model creation failed."
                )

            # ----------------------------------
            # SAVE GLOBAL REFERENCES
            # ----------------------------------

            MODEL = model
            TOKENIZER = tokenizer
            EMBEDDING_MODEL = embedding_model

            _NLP_FAILED = False

            # ----------------------------------
            # RESET EMBEDDING CACHE
            # ----------------------------------

            with _EMB_LOCK:

                _EMB_CACHE.update(
                    key=None,
                    ids=None,
                    matrix=None
                )

            # ----------------------------------
            # SUCCESS
            # ----------------------------------

            print(
                "✅ CNN + NLP initialization "
                "completed successfully."
            )

            print(
                "✅ Main model, tokenizer, "
                "and embedding model are ready."
            )

            return True

        # ----------------------------------
        # ERROR HANDLING
        # ----------------------------------

        except Exception as exc:

            MODEL = None
            TOKENIZER = None
            EMBEDDING_MODEL = None

            _NLP_FAILED = True

            print(
                f"❌ NLP model initialization failed: {exc}"
            )

            return False
def get_content_embeddings(descriptions, batch_size=32):
    """Return a 2-D array (n_texts, embedding_dim): mean of word embeddings per text."""
    if not text_model_ready():
        return np.zeros((0, 0), dtype=np.float32)

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = TOKENIZER.texts_to_sequences([d or '' for d in descriptions])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    chunks = []
    for start in range(0, len(padded), batch_size):
        batch = padded[start:start + batch_size]
        emb = np.asarray(EMBEDDING_MODEL.predict(batch, verbose=0))
        if emb.ndim == 3:  # (batch, seq_len, dim) -> mask padding and average
            mask = (batch != 0).astype(np.float32)[:, :, None]
            emb = (emb * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        chunks.append(emb.astype(np.float32))

    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def _cosine(vector, matrix):
    v = vector / (np.linalg.norm(vector) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ v


# =========================================================
# RECOMMENDERS
# =========================================================
class ALSRecommender:
    """Memory-bounded SVD collaborative filtering (matrix factorisation)."""

    MAX_USERS = 200
    MAX_ITEMS = 500
    MAX_USER_ITEMS = 100
    MIN_RATINGS = 5
    MAX_FACTORS = 20
    CACHE_SECONDS = 300
    MAX_CACHE_USERS = 256

    _cache = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _get_cached(cls, user_id):
        now = time.monotonic()
        with cls._cache_lock:
            cached = cls._cache.get(user_id)
            if cached:
                stamp, predictions = cached
                if now - stamp < cls.CACHE_SECONDS:
                    return predictions.copy()
                cls._cache.pop(user_id, None)
        return None

    @classmethod
    def _save_cache(cls, user_id, predictions):
        now = time.monotonic()
        with cls._cache_lock:
            for uid in [u for u, v in cls._cache.items() if now - v[0] >= cls.CACHE_SECONDS]:
                cls._cache.pop(uid, None)
            if len(cls._cache) >= cls.MAX_CACHE_USERS:
                oldest = min(cls._cache, key=lambda u: cls._cache[u][0])
                cls._cache.pop(oldest, None)
            cls._cache[user_id] = (now, predictions.copy())

    def predict(self, user_id, all_content_ids=None):
        if user_id is None:
            return {}

        result = self._get_cached(user_id)
        if result is None:
            result = self._compute(user_id)
            if result is None:      # error -> don't cache
                return {}
            self._save_cache(user_id, result)

        if all_content_ids is not None:
            allowed = set(all_content_ids)
            return {cid: s for cid, s in result.items() if cid in allowed}
        return result

    def _compute(self, user_id):
        conn = None
        try:
            conn = get_connection()
            if conn is None:
                return None

            # NOTE: tuple cursor on purpose (default cursor is DictCursor)
            with conn.cursor(pymysql.cursors.Cursor) as cursor:
                cursor.execute(
                    "SELECT user_id FROM ratings GROUP BY user_id "
                    "ORDER BY COUNT(*) DESC LIMIT %s", (self.MAX_USERS,))
                user_ids = [row[0] for row in cursor.fetchall()]
                if user_id not in user_ids:
                    user_ids = user_ids[:self.MAX_USERS - 1]
                    user_ids.append(user_id)

                cursor.execute(
                    "SELECT content_id FROM ratings GROUP BY content_id "
                    "ORDER BY COUNT(*) DESC LIMIT %s", (self.MAX_ITEMS,))
                content_ids = [row[0] for row in cursor.fetchall()]

                cursor.execute(
                    "SELECT content_id FROM ratings WHERE user_id=%s LIMIT %s",
                    (user_id, self.MAX_USER_ITEMS))
                known = set(content_ids)
                for (cid,) in cursor.fetchall():
                    if cid not in known:
                        content_ids.append(cid)
                        known.add(cid)

                if not content_ids:
                    return {}

                u_ph = ",".join(["%s"] * len(user_ids))
                c_ph = ",".join(["%s"] * len(content_ids))
                cursor.execute(
                    f"""
                    SELECT user_id, content_id, AVG(rating) AS rating
                    FROM ratings
                    WHERE user_id IN ({u_ph}) AND content_id IN ({c_ph})
                    GROUP BY user_id, content_id
                    """,
                    tuple(user_ids) + tuple(content_ids))
                rows = cursor.fetchall()

            if len(rows) < self.MIN_RATINGS:
                return {}

            user_index = {uid: i for i, uid in enumerate(user_ids)}
            item_index = {cid: i for i, cid in enumerate(content_ids)}
            matrix = np.zeros((len(user_ids), len(content_ids)), dtype=np.float32)

            for r_user, r_content, rating in rows:
                ri, ci = user_index.get(r_user), item_index.get(r_content)
                if ri is not None and ci is not None and rating is not None:
                    matrix[ri, ci] = float(rating)

            if np.count_nonzero(matrix) < self.MIN_RATINGS:
                return {}

            me = user_index[user_id]
            rated_mask = matrix > 0
            if not rated_mask[me].any():       # user has no ratings -> nothing to predict
                return {}

            counts = rated_mask.sum(axis=1)
            user_means = np.divide(
                matrix.sum(axis=1), counts,
                out=np.zeros(matrix.shape[0], dtype=np.float32),
                where=counts > 0)

            demeaned = np.where(rated_mask, matrix - user_means.reshape(-1, 1), 0).astype(np.float32)

            k = min(self.MAX_FACTORS, min(demeaned.shape) - 1)
            if k < 1:
                return {}

            U, sigma, Vt = np.linalg.svd(demeaned, full_matrices=False)
            predicted = (U[:, :k] * sigma[:k]) @ Vt[:k, :] + user_means.reshape(-1, 1)
            mine = predicted[me]

            result = {}
            for idx, cid in enumerate(content_ids):
                if matrix[me, idx] > 0:       # skip already-rated
                    continue
                score = float(mine[idx])
                if np.isfinite(score):
                    result[cid] = float(np.clip(score, 0.0, 5.0))
            return result

        except Exception as e:
            print("ALS Recommendation Error:", e)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


class DemographicPSORecommender:
    """Demographic similarity (age / gender / genre overlap)."""

    def get_similar_users(self, user_id):
        conn = get_connection()
        if conn is None:
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT age, gender, fav_genre FROM users WHERE id=%s", (user_id,))
                target = cursor.fetchone()
                if not target:
                    return []
                cursor.execute("SELECT id, age, gender, fav_genre FROM users WHERE id != %s", (user_id,))
                others = cursor.fetchall()
        finally:
            conn.close()

        def genres_of(value):
            return {g.strip() for g in (value or '').split(',') if g.strip()}

        target_age = int(target.get('age') or 25)
        target_genres = genres_of(target.get('fav_genre'))

        scored = []
        for u in others:
            score = 1 / (1 + abs(target_age - int(u.get('age') or 25)) * 0.1)
            if target.get('gender') == u.get('gender'):
                score += 0.5
            u_genres = genres_of(u.get('fav_genre'))
            union = len(target_genres | u_genres)
            if union:
                score += (len(target_genres & u_genres) / union) * 2.0
            scored.append((u['id'], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    def predict(self, user_id):
        similar = self.get_similar_users(user_id)
        if not similar:
            return {}

        ids = [u[0] for u in similar]
        conn = get_connection()
        if conn is None:
            return {}
        scores = {}
        try:
            ph = ','.join(['%s'] * len(ids))
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT content_id, rating FROM ratings WHERE user_id IN ({ph}) AND rating >= 4",
                    tuple(ids))
                for r in cursor.fetchall():
                    scores[r['content_id']] = scores.get(r['content_id'], 0) + float(r['rating'])
            return scores
        finally:
            conn.close()


class SocialRecommender:
    """What the people you follow rated / liked."""

    def predict(self, user_id):
        conn = get_connection()
        if conn is None:
            return {}
        scores = {}
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT followed_id FROM social_graph WHERE follower_id=%s", (user_id,))
                following = [row['followed_id'] for row in cursor.fetchall()]
                if not following:
                    return {}

                ph = ','.join(['%s'] * len(following))
                cursor.execute(
                    f"SELECT content_id, rating FROM ratings WHERE user_id IN ({ph})", tuple(following))
                for r in cursor.fetchall():
                    scores[r['content_id']] = scores.get(r['content_id'], 0) + float(r['rating'] or 0)

                cursor.execute(
                    f"SELECT content_id FROM likes WHERE user_id IN ({ph})", tuple(following))
                for r in cursor.fetchall():
                    scores[r['content_id']] = scores.get(r['content_id'], 0) + 5.0
            return scores
        finally:
            conn.close()


class HybridEngine:
    """Weighted fusion of ALS + demographic + text + social scores."""

    def __init__(self):
        self.als = ALSRecommender()
        self.demo = DemographicPSORecommender()
        self.social = SocialRecommender()
        self.W_ALS = 0.35
        self.W_DEMO = 0.20
        self.W_TEXT = 0.25
        self.W_SOCIAL = 0.20

    @staticmethod
    def normalize(scores):
        if not scores:
            return {}
        clipped = {k: max(float(v), 0.0) for k, v in scores.items()}
        top = max(clipped.values())
        if top <= 0:
            return {}
        return {k: v / top for k, v in clipped.items()}

    def _safe(self, name, func, *args):
        try:
            return self.normalize(func(*args))
        except Exception as exc:
            print(f"{name} engine error:", exc)
            return {}

    @staticmethod
    def _load_catalog(user_id):
        conn = get_connection()
        if conn is None:
            return [], set()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, story FROM contents WHERE is_kids = 0 OR is_kids IS NULL")
                rows = cursor.fetchall()
                cursor.execute("SELECT content_id FROM user_history WHERE user_id=%s", (user_id,))
                watched = {r['content_id'] for r in cursor.fetchall()}
            return rows, watched
        finally:
            conn.close()

    @staticmethod
    def _text_scores(last_desc, rows):
        if not last_desc or not rows or not text_model_ready():
            return {}

        ids = [r['id'] for r in rows]
        stories = [r['story'] or '' for r in rows]
        key = (len(ids), sum(ids), sum(len(s) for s in stories))

        with _EMB_LOCK:
            if _EMB_CACHE['key'] != key:
                matrix = get_content_embeddings(stories)
                if matrix.size == 0:
                    return {}
                _EMB_CACHE.update(key=key, ids=ids, matrix=matrix)
            matrix, cache_ids = _EMB_CACHE['matrix'], _EMB_CACHE['ids']

        vec = get_content_embeddings([last_desc])
        if vec.size == 0:
            return {}
        sims = _cosine(vec[0], matrix)
        return {cid: float(s) for cid, s in zip(cache_ids, sims)}

    def get_hybrid_recommendations(self, user_id, last_watched_desc=None, exclude_ids=None, limit=10):
        rows, watched = self._load_catalog(user_id)
        allowed = {r['id'] for r in rows}
        skip = set(watched) | set(exclude_ids or [])

        scores_als = self._safe('ALS', self.als.predict, user_id, None)
        scores_demo = self._safe('Demographic', self.demo.predict, user_id)
        scores_social = self._safe('Social', self.social.predict, user_id)
        scores_text = self._safe('Text', self._text_scores, last_watched_desc, rows)

        keys = (set(scores_als) | set(scores_demo) | set(scores_social) | set(scores_text)) & allowed
        keys -= skip

        final = {}
        for cid in keys:
            final[cid] = (self.W_ALS * scores_als.get(cid, 0)
                          + self.W_DEMO * scores_demo.get(cid, 0)
                          + self.W_SOCIAL * scores_social.get(cid, 0)
                          + self.W_TEXT * scores_text.get(cid, 0))

        return sorted(final, key=final.get, reverse=True)[:limit]


# =========================================================
# SHARED HELPERS / DECORATORS
# =========================================================
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'loggedin' not in session or session.get('user_type') != 'admin':
            flash('Please log in as an admin to access this page.', 'danger')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def resolve_profile_image(uploaded, gender):
    default = 'f_profile.png' if (gender or '').strip().lower() == 'female' else 'm_profile.png'
    if uploaded:
        name = os.path.basename(uploaded.strip())
        if os.path.isfile(os.path.join(current_app.static_folder, 'uploads', name)):
            return 'uploads/' + name
    return default


def save_image_file(file_storage, prefix=''):
    """Validate + save an uploaded image under a unique name. Returns (name, path) or (None, None)."""
    if not file_storage or not file_storage.filename:
        return None, None
    if not allowed_file(file_storage.filename):
        raise ValueError('Unsupported image format. Use png, jpg, jpeg or gif.')
    safe = secure_filename(file_storage.filename)
    if not safe:
        raise ValueError('Invalid filename.')
    ext = os.path.splitext(safe)[1].lower()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    name = f"{prefix}{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    file_storage.save(path)
    return name, path


def extract_youtube_id(value):
    """Return an 11-char YouTube id from a URL (or a raw id), else ''."""
    value = (value or '').strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        return value
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or '').lower()
        if host.startswith('www.'):
            host = host[4:]
        video_id = ''
        if parsed.scheme in ('http', 'https'):
            if host in ('youtube.com', 'm.youtube.com', 'music.youtube.com'):
                if parsed.path == '/watch':
                    video_id = parse_qs(parsed.query).get('v', [''])[0]
                elif parsed.path.startswith(('/embed/', '/shorts/', '/live/')):
                    video_id = parsed.path.split('/')[2]
            elif host == 'youtu.be':
                video_id = parsed.path.strip('/').split('/')[0]
        return video_id if re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id) else ''
    except (ValueError, IndexError, AttributeError):
        return ''


def record_history(cursor, user_id, content_id):
    cursor.execute("SELECT id FROM user_history WHERE user_id=%s AND content_id=%s",
                   (user_id, content_id))
    if cursor.fetchone():
        cursor.execute("UPDATE user_history SET watched_at = NOW() WHERE user_id=%s AND content_id=%s",
                       (user_id, content_id))
    else:
        cursor.execute("INSERT INTO user_history (user_id, content_id, watched_at) VALUES (%s, %s, NOW())",
                       (user_id, content_id))


def touch_continue_watching(cursor, user_id, content_id, progress=None):
    """Insert/update continue_watching without relying on a unique index.
    progress=None keeps the existing progress (or 0 for a new row)."""
    cursor.execute("SELECT progress FROM continue_watching WHERE user_id=%s AND content_id=%s",
                   (user_id, content_id))
    row = cursor.fetchone()
    if row:
        if progress is None:
            cursor.execute("UPDATE continue_watching SET last_watched = NOW() "
                           "WHERE user_id=%s AND content_id=%s", (user_id, content_id))
        else:
            cursor.execute("UPDATE continue_watching SET progress=%s, last_watched = NOW() "
                           "WHERE user_id=%s AND content_id=%s", (progress, user_id, content_id))
        return row['progress']
    cursor.execute("INSERT INTO continue_watching (user_id, content_id, progress, last_watched) "
                   "VALUES (%s, %s, %s, NOW())", (user_id, content_id, progress or 0))
    return progress or 0


def fetch_recommendations(cursor, user_id, story, current_id=None, limit=12):
    """Hybrid recommendations -> list of content rows (ordered), [] on any failure."""
    try:
        exclude = [current_id] if current_id else []
        rec_ids = hybrid_engine.get_hybrid_recommendations(
            user_id, story, exclude_ids=exclude, limit=limit)
        if not rec_ids:
            return []
        ph = ",".join(["%s"] * len(rec_ids))
        cursor.execute(f"SELECT * FROM contents WHERE id IN ({ph}) AND (is_kids = 0 OR is_kids IS NULL)",
                       tuple(rec_ids))
        data = cursor.fetchall()
        data.sort(key=lambda x: rec_ids.index(x['id']))
        return data
    except Exception as e:
        print("Hybrid Engine Error:", e)
        return []


# =========================================================
# INITIALISE
# =========================================================
setup_database()
hybrid_engine = HybridEngine()

if ENABLE_TEXT_MODEL:
    # Load TensorFlow in the background so the web server boots (and passes
    # Render's port check) immediately.
    threading.Thread(target=load_nlp_model, daemon=True).start()


@app.route('/healthz')
def healthz():
    return 'ok', 200


# =========================================================
# ROUTES: HOME
# =========================================================
@app.route('/')
def index():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        base = """
            SELECT * FROM contents
            WHERE content_type = %s
              AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)
            ORDER BY views DESC, hearts DESC
            LIMIT 10
        """
        cursor.execute(base, ('movie',))
        trending_movies = cursor.fetchall()
        cursor.execute(base, ('series',))
        trending_series = cursor.fetchall()
        cursor.close()
        return render_template("index.html",
                               trending_movies=trending_movies,
                               trending_series=trending_series)
    except Exception as e:
        flash(f"Database Error: {str(e)}", "danger")
        return render_template("index.html", trending_movies=[], trending_series=[])
    finally:
        if conn:
            conn.close()


# =========================================================
# ROUTES: ADMIN
# =========================================================
@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    conn = get_connection()
    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('login'))
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM users")
            user_count = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) AS count FROM contents")
            content_count = cursor.fetchone()['count']

            cursor.execute("SELECT * FROM model_status ORDER BY id DESC LIMIT 1")
            model = cursor.fetchone()

            model_trained, last_trained, accuracy = False, None, None
            if model:
                model_trained = True
                trained_at = model.get('trained_at')
                if isinstance(trained_at, str):
                    last_trained = trained_at
                elif trained_at:
                    last_trained = trained_at.strftime("%d %b %Y, %I:%M %p")
                accuracy = model.get('accuracy')

            cursor.execute("SELECT * FROM contents ORDER BY id DESC")
            contents = cursor.fetchall()

        return render_template('admin_dashboard.html',
                               contents=contents, user_count=user_count,
                               content_count=content_count, model_trained=model_trained,
                               last_trained=last_trained, accuracy=accuracy)
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('login'))
    finally:
        conn.close()


@app.route('/admin/users')
@admin_required
def admin_users():
    search = request.args.get('search', '').strip()
    page = max(request.args.get('page', 1, type=int), 1)
    per_page = 10
    offset = (page - 1) * per_page

    conn = get_connection()
    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('login'))

    try:
        cursor = conn.cursor()
        where_sql, params = "", []
        if search:
            where_sql = " WHERE (username LIKE %s OR fav_genre LIKE %s OR fav_language LIKE %s)"
            params.extend([f"%{search}%"] * 3)

        cursor.execute(f"SELECT COUNT(*) as total FROM users {where_sql}", params)
        total_users = cursor.fetchone()['total']

        cursor.execute(f"""
            SELECT id, username, fav_genre, fav_language, gender, profile_image,
                   COALESCE(is_online, 0) AS is_online,
                   COALESCE(watch_time, 0) AS watch_time
            FROM users
            {where_sql}
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, params + [per_page, offset])
        users = cursor.fetchall()

        total_pages = (total_users + per_page - 1) // per_page
        return render_template('admin_users.html', users=users, page=page,
                               total_pages=total_pages, search=search)
    except Exception as e:
        flash(f'Error loading users: {e}', 'danger')
        return redirect(url_for('admin_dashboard'))
    finally:
        conn.close()


@app.route('/admin/users/wordcloud/<int:user_id>')
@admin_required
def user_wordcloud(user_id):
    conn = get_connection()
    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('login'))

    wordcloud_img = None
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, gender, profile_image, COALESCE(is_online,0) as is_online
            FROM users WHERE id=%s
        """, (user_id,))
        user = cursor.fetchone()
        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        if user.get('profile_image'):
            profile_image_url = url_for('static', filename=f'uploads/{user["profile_image"]}')
        elif (user.get('gender') or '').lower() == 'male':
            profile_image_url = url_for('static', filename='m_profile.png')
        else:
            profile_image_url = url_for('static', filename='f_profile.png')

        cursor.execute("""
            SELECT c.story FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
        """, (user_id,))
        text = " ".join(s['story'] for s in cursor.fetchall() if s.get('story'))

        if text.strip():
            img_dir = os.path.join(app.root_path, 'static', 'img')
            os.makedirs(img_dir, exist_ok=True)
            WordCloud(width=800, height=400, background_color='black',
                      colormap='Reds').generate(text).to_file(
                os.path.join(img_dir, f'wordcloud_{user_id}.png'))
            wordcloud_img = url_for('static', filename=f'img/wordcloud_{user_id}.png')

        cursor.execute("""
            SELECT c.title, c.thumbnail, COUNT(*) as views
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.id, c.title, c.thumbnail
            ORDER BY views DESC LIMIT 5
        """, (user_id,))
        watch_time_data = cursor.fetchall()

        cursor.execute("""
            SELECT c.title, c.thumbnail
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            ORDER BY uh.watched_at DESC LIMIT 6
        """, (user_id,))
        recent_contents = cursor.fetchall()

        return render_template('user_wordcloud.html', user=user, wordcloud_img=wordcloud_img,
                               profile_image_url=profile_image_url,
                               watch_time_data=watch_time_data,
                               recent_contents=recent_contents)
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        return redirect(url_for('admin_users'))
    finally:
        conn.close()


CONTENT_TEXT_FIELDS = [
    'hero', 'hero_caste', 'hero2', 'hero2_caste', 'hero3', 'hero3_caste',
    'heroine', 'heroine_caste', 'heroine2', 'heroine2_caste', 'heroine3', 'heroine3_caste',
    'director', 'comedian', 'comedian_caste', 'producer', 'producer_caste',
]
CONTENT_IMAGE_FIELDS = [
    'hero_image', 'hero2_image', 'hero3_image',
    'heroine_image', 'heroine2_image', 'heroine3_image',
    'director_image', 'comedian_image', 'producer_image',
]


@app.route('/admin/upload', methods=['GET', 'POST'])
@admin_required
def upload_content():
    if request.method == 'GET':
        return render_template('upload_content.html')

    conn = None
    cursor = None
    saved_files = []

    try:
        title = request.form.get('title', '').strip()
        story = request.form.get('story', '').strip()
        content_type = request.form.get('content_type', '').strip()
        if not title or not story or not content_type:
            flash('Title, story and content type are required.', 'danger')
            return redirect(request.url)

        video_id = extract_youtube_id(request.form.get('youtube_url', ''))
        if not video_id:
            flash('Invalid YouTube URL. Please enter a valid video link.', 'danger')
            return redirect(request.url)

        def optional_int(field, default=None):
            value = request.form.get(field, '').strip()
            return default if not value else int(value)

        try:
            release_year = optional_int('release_year')
            trending_rank = optional_int('trending_rank', 0)
            deepflix_rank = optional_int('deepflix_rank', 0)
        except ValueError:
            flash('Year and rank fields must be whole numbers.', 'danger')
            return redirect(request.url)

        if trending_rank < 0 or deepflix_rank < 0:
            flash('Rank values cannot be negative.', 'danger')
            return redirect(request.url)

        record = {
            'title': title,
            'story': story,
            'language': request.form.get('language', '').strip(),
            'release_year': release_year,
            'content_type': content_type,
            'genre': request.form.get('genre', '').strip(),
            'youtube_video_id': video_id,
            'backdrop_url': request.form.get('backdrop_url', '').strip() or None,
            'trending_rank': trending_rank,
            'deepflix_rank': deepflix_rank,
            'is_90s': 1 if request.form.get('is_90s') else 0,
            'is_kids': 1 if request.form.get('is_kids') else 0,
        }
        for field in CONTENT_TEXT_FIELDS:
            record[field] = request.form.get(field, '').strip()

        name, path = save_image_file(request.files.get('thumbnail'))
        if not name:
            raise ValueError('Please select a thumbnail image.')
        saved_files.append(path)
        record['thumbnail'] = name

        for field in CONTENT_IMAGE_FIELDS:
            name, path = save_image_file(request.files.get(field))
            record[field] = name or ''
            if path:
                saved_files.append(path)

        conn = get_connection()
        if conn is None:
            raise RuntimeError('Database connection could not be created.')
        cursor = conn.cursor()

        if deepflix_rank > 0:
            cursor.execute("UPDATE contents SET deepflix_rank = 0 WHERE deepflix_rank = %s",
                           (deepflix_rank,))

        columns = list(record)  # hard-coded keys only
        cursor.execute(
            f"INSERT INTO contents ({', '.join(columns)}) "
            f"VALUES ({', '.join(['%s'] * len(columns))})",
            tuple(record[c] for c in columns))
        content_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO notifications (title, message, content_id) VALUES (%s, %s, %s)",
            ('New Content Added', f'{title} is now available to watch!', content_id))

        conn.commit()
        saved_files.clear()
        flash('Content uploaded successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    except ValueError as e:
        if conn:
            conn.rollback()
        flash(str(e), 'danger')
        return redirect(request.url)
    except Exception:
        if conn:
            conn.rollback()
        app.logger.exception('Content upload failed')
        flash('Upload failed. Please check the application logs and database connection.', 'danger')
        return redirect(request.url)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        for path in saved_files:  # remove files if DB insert failed
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                app.logger.warning('Could not remove temporary upload: %s', path)


@app.route('/admin/upload_csv', methods=['GET', 'POST'])
@admin_required
def upload_csv():
    if request.method == 'GET':
        return render_template('upload_csv.html')

    if 'csv_file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(request.url)
    file = request.files['csv_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    if not file.filename.lower().endswith('.csv'):
        flash('Only CSV allowed', 'danger')
        return redirect(request.url)

    conn = None
    cursor = None
    try:
        import pandas as pd

        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8-sig')))

        def val(row, key):
            v = row.get(key)
            return None if v is None or pd.isna(v) else v

        def as_int(row, key, default=0):
            v = val(row, key)
            return default if v is None else int(v)

        def as_flag(row, key):
            return 1 if str(val(row, key)).strip().lower() in ('true', '1', 'yes', 'y') else 0

        conn = get_connection()
        if conn is None:
            raise RuntimeError('Database connection failed')
        cursor = conn.cursor()

        success = failed = 0
        text_cols = ['title', 'story', 'language', 'content_type', 'genre', 'thumbnail',
                     'backdrop_url'] + CONTENT_TEXT_FIELDS + CONTENT_IMAGE_FIELDS

        for index, row in df.iterrows():
            try:
                youtube_url = str(val(row, 'youtube_url') or '')
                video_id = extract_youtube_id(youtube_url)
                if not video_id or not val(row, 'title') or not val(row, 'story') \
                        or not val(row, 'content_type'):
                    failed += 1
                    continue

                record = {c: val(row, c) for c in text_cols}
                record['release_year'] = as_int(row, 'release_year', None)
                record['youtube_video_id'] = video_id
                record['trending_rank'] = as_int(row, 'trending_rank')
                record['deepflix_rank'] = as_int(row, 'deepflix_rank')
                record['is_90s'] = as_flag(row, 'is_90s')
                record['is_kids'] = as_flag(row, 'is_kids')

                if record['deepflix_rank'] > 0:
                    cursor.execute("UPDATE contents SET deepflix_rank = 0 WHERE deepflix_rank = %s",
                                   (record['deepflix_rank'],))

                columns = list(record)
                cursor.execute(
                    f"INSERT INTO contents ({', '.join(columns)}) "
                    f"VALUES ({', '.join(['%s'] * len(columns))})",
                    tuple(record[c] for c in columns))
                content_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO notifications (title, message, content_id) VALUES (%s,%s,%s)",
                    ("New Content Added", f"{record['title']} is now available!", content_id))
                success += 1
            except Exception as row_error:
                print(f"CSV row {index} error:", row_error)
                failed += 1

        conn.commit()
        flash(f"CSV Upload Done ✅ {success} success, {failed} failed", "success")
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        if conn:
            conn.rollback()
        flash(f"Error: {str(e)}", "danger")
        return redirect(request.url)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ==========================================
# BACKGROUND MODEL TRAINING
# ==========================================

def run_training_background(script_path):

    print("DeepFlixAI background training started.", flush=True)

    try:

        # Run training script without blocking the browser request
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False
        )

        if result.returncode != 0:

            print(
                f"Training failed with exit code: {result.returncode}",
                flush=True
            )

            return

        # ----------------------------------
        # READ REAL MODEL METRICS
        # ----------------------------------

        accuracy = None

        try:

            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            series = (
                metrics.get('val_accuracy')
                or metrics.get('accuracy')
                or []
            )

            if series:

                accuracy = float(series[-1])

                if accuracy <= 1:
                    accuracy *= 100

                accuracy = round(accuracy, 2)

        except Exception as e:

            print(
                f"Could not read training metrics: {e}",
                flush=True
            )

        # ----------------------------------
        # SAVE TRAINING STATUS
        # ----------------------------------

        conn = get_connection()

        if not conn:

            print(
                "Training completed but database connection "
                "failed while saving status.",
                flush=True
            )

            return

        try:

            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO model_status (trained_at, accuracy)
                    VALUES (%s, %s)
                    """,
                    (datetime.now(), accuracy)
                )

            conn.commit()

        finally:

            conn.close()

        # ----------------------------------
        # RELOAD TRAINED NLP MODEL
        # ----------------------------------

        load_nlp_model(force=True)

        print(
            f"Training successfully completed. Accuracy: {accuracy}",
            flush=True
        )

    except Exception as e:

        print(
            f"Background training error: {e}",
            flush=True
        )

        traceback.print_exc()


# ==========================================
# ADMIN TRAIN MODEL
# ==========================================

@app.route('/admin/train_model', methods=['POST'])
@admin_required
def train_model():

    script_path = os.path.join(
        BASE_DIR,
        'train_model.py'
    )

    if not os.path.isfile(script_path):

        flash(
            'Training script not found.',
            'danger'
        )

        return redirect(url_for('admin_dashboard'))

    try:

        # ----------------------------------
        # START BACKGROUND TRAINING
        # ----------------------------------

        training_thread = threading.Thread(

            target=run_training_background,

            args=(script_path,),

            daemon=True

        )

        training_thread.start()

        flash(
            'Model training started in background. '
            'Please check server logs for progress.',
            'success'
        )

    except Exception as e:

        print(
            f"Could not start training: {e}",
            flush=True
        )

        flash(
            f'Unable to start training: {e}',
            'danger'
        )

    return redirect(url_for('admin_dashboard'))
@app.route('/admin/analysis')
@admin_required
def admin_analysis():
    def existing(rel):
        return url_for('static', filename=rel) if os.path.exists(
            os.path.join(app.root_path, 'static', rel)) else None

    return render_template('analysis.html',
                           accuracy_loss_img=existing('img/accuracy_loss.png'),
                           confusion_matrix_img=existing('img/confusion_matrix.png'))


@app.route('/admin/edit/<int:content_id>', methods=['GET', 'POST'])
@admin_required
def edit_content(content_id):
    conn = None
    new_files = []
    try:
        conn = get_connection()
        if not conn:
            flash("Database connection failed!", "danger")
            return redirect(url_for('admin_dashboard'))

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
        content = cursor.fetchone()
        if not content:
            flash("Content not found!", "danger")
            return redirect(url_for('admin_dashboard'))

        if request.method == 'GET':
            return render_template('edit_content.html', content=content)

        form = request.form
        release_year = int(form.get('release_year')) if form.get('release_year') else None
        trending_rank = int(form.get('trending_rank') or 0)
        deepflix_rank = int(form.get('deepflix_rank') or 0)

        youtube_input = form.get('youtube_video_id', '').strip()
        youtube_video_id = (extract_youtube_id(youtube_input) or youtube_input) \
            if youtube_input else content.get('youtube_video_id')

        upload_date = form.get('upload_date') or content.get('upload_date')

        fields = {
            'title': form.get('title'),
            'story': form.get('story'),
            'language': form.get('language'),
            'release_year': release_year,
            'content_type': form.get('content_type'),
            'genre': form.get('genre'),
            'hero': form.get('hero'), 'hero2': form.get('hero2'), 'hero3': form.get('hero3'),
            'heroine': form.get('heroine'), 'heroine2': form.get('heroine2'),
            'heroine3': form.get('heroine3'),
            'director': form.get('director'), 'comedian': form.get('comedian'),
            'producer': form.get('producer'),
            'upload_date': upload_date,
            'trending_rank': trending_rank,
            'deepflix_rank': deepflix_rank,
            'youtube_video_id': youtube_video_id,
            'is_90s': 1 if form.get('is_90s') else 0,
            'backdrop_url': form.get('backdrop_url', '').strip() or content.get('backdrop_url'),
        }

        for field in ['thumbnail'] + CONTENT_IMAGE_FIELDS:
            name, path = save_image_file(request.files.get(field))
            if name:
                fields[field] = name
                new_files.append(path)

        if deepflix_rank > 0:
            cursor.execute("UPDATE contents SET deepflix_rank=0 WHERE deepflix_rank=%s AND id!=%s",
                           (deepflix_rank, content_id))

        columns = list(fields)  # hard-coded keys only
        cursor.execute(
            f"UPDATE contents SET {', '.join(c + '=%s' for c in columns)} WHERE id=%s",
            tuple(fields[c] for c in columns) + (content_id,))
        conn.commit()
        new_files.clear()

        flash("Content updated successfully!", "success")
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        print("EDIT ERROR:", e)
        if conn:
            conn.rollback()
        flash(f"Error updating content: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))
    finally:
        if conn:
            conn.close()
        for path in new_files:
            try:
                os.remove(path)
            except OSError:
                pass


@app.route('/admin/delete/<int:content_id>')
@admin_required
def delete_content(content_id):
    conn = get_connection()
    if not conn:
        flash("Database connection failed!", "danger")
        return redirect(url_for('admin_dashboard'))
    try:
        with conn.cursor() as cursor:
            # remove rows that reference this content first (foreign keys)
            for table in ('ratings', 'reviews', 'likes', 'user_hearts', 'user_history',
                          'continue_watching', 'notifications'):
                safe_delete(cursor, table, 'content_id', content_id)
            cursor.execute("DELETE FROM contents WHERE id=%s", (content_id,))
        conn.commit()
        flash("Content deleted successfully!", "danger")
    except Exception as e:
        conn.rollback()
        flash(f"Delete failed: {str(e)}", "danger")
    finally:
        conn.close()
    return redirect(url_for('admin_dashboard'))


# =========================================================
# ROUTES: AUTH + PROFILES
# =========================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_type = request.form.get('user_type', 'user')

        conn = get_connection()
        if not conn:
            flash("Database connection error. Try again in a moment.", "danger")
            return render_template("login.html")

        try:
            with conn.cursor() as cursor:
                table = "admins" if user_type == "admin" else "users"
                cursor.execute(f"SELECT * FROM {table} WHERE username=%s", (username,))
                user = cursor.fetchone()

            if user and verify_password_hash(user['password'], password):
                session.clear()
                session['loggedin'] = True
                session['user_id'] = user['id']
                session['username'] = user['username']

                if table == "admins":
                    session['user_type'] = "admin"
                    return redirect(url_for('admin_dashboard'))

                session['user_type'] = "user"
                return redirect(url_for('watch_based'))

            flash("Invalid username or password", "danger")
        finally:
            conn.close()

    return render_template("login.html")


@app.route('/google-login', methods=['POST'])
def google_login():
    """Verify a Google Identity Services ID token and sign in/create the user."""
    if not GOOGLE_CLIENT_ID:
        return jsonify({'success': False, 'message': 'Google sign-in is not configured on the server.'}), 503

    payload = request.get_json(silent=True) or {}
    credential = (payload.get('credential') or '').strip()
    if not credential:
        return jsonify({'success': False, 'message': 'Google credential is missing.'}), 400

    try:
        # Import lazily so the rest of DeepFlix still starts if Google auth is unused.
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        claims = id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        # verify_oauth2_token validates signature, issuer, expiry and audience.
        google_id = claims.get('sub')
        email = (claims.get('email') or '').strip().lower()
        email_verified = claims.get('email_verified') is True
        display_name = (claims.get('name') or email.split('@')[0]).strip()[:100]
        if not google_id or not email or not email_verified:
            return jsonify({'success': False, 'message': 'Google account email is not verified.'}), 401
    except ImportError:
        app.logger.exception('Google authentication dependencies are missing')
        return jsonify({'success': False, 'message': 'Server dependency missing: install google-auth.'}), 500
    except Exception:
        app.logger.exception('Google ID token verification failed')
        return jsonify({'success': False, 'message': 'Google sign-in verification failed. Please try again.'}), 401

    conn = get_connection()
    if conn is None:
        return jsonify({'success': False, 'message': 'Database connection failed.'}), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE google_id=%s LIMIT 1", (google_id,))
            user = cursor.fetchone()

            if not user:
                # Link an existing account by its verified email instead of creating a duplicate.
                cursor.execute("SELECT * FROM users WHERE LOWER(email)= %s LIMIT 1", (email,))
                user = cursor.fetchone()

            if user:
                cursor.execute("UPDATE users SET google_id=%s, email=%s, auth_provider='google' WHERE id=%s",
                               (google_id, email, user['id']))
                user_id = user['id']
                username = user.get('username') or display_name
            else:
                # Keep usernames unique even when two Google accounts share a display name.
                base_username = re.sub(r'[^A-Za-z0-9_.-]+', '_', email.split('@')[0]).strip('_.-') or 'google_user'
                username = base_username[:80]
                suffix = 1
                while True:
                    cursor.execute("SELECT id FROM users WHERE username=%s LIMIT 1", (username,))
                    if not cursor.fetchone():
                        break
                    suffix += 1
                    username = f"{base_username[:70]}_{suffix}"

                # The random unusable password keeps the existing NOT NULL schema compatible.
                random_password = generate_password_hash(uuid.uuid4().hex + uuid.uuid4().hex, method='pbkdf2:sha256')
                cursor.execute("""
                    INSERT INTO users
                        (username, password, fav_genre, fav_language, age, gender, email, google_id, auth_provider)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'google')
                """, (username, random_password, '', '', 25, 'Other', email, google_id))
                user_id = cursor.lastrowid

        conn.commit()
        session.clear()
        session['loggedin'] = True
        session['user_id'] = user_id
        session['username'] = username
        session['user_type'] = 'user'
        session['auth_provider'] = 'google'
        return jsonify({'success': True, 'redirect_url': url_for('watch_based')})
    except pymysql.err.IntegrityError:
        conn.rollback()
        app.logger.exception('Google account already linked or username conflicts')
        return jsonify({'success': False, 'message': 'This Google account or username is already linked. Please contact support.'}), 409
    except Exception:
        conn.rollback()
        app.logger.exception('Google login database operation failed')
        return jsonify({'success': False, 'message': 'Unable to sign in with Google right now.'}), 500
    finally:
        conn.close()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        fav_genre = request.form.get('fav_genre', '')
        fav_language = request.form.get('fav_language', '')
        gender = request.form.get('gender', 'Other')
        try:
            age = int(request.form.get('age', ''))
        except ValueError:
            flash('Please enter a valid age.', 'danger')
            return redirect(url_for('register'))

        if age < 18:
            flash('Age Restricted! Only 18+ users can register.', 'danger')
            return redirect(url_for('register'))
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('register'))

        conn = get_connection()
        if not conn:
            flash('Database connection error.', 'danger')
            return redirect(url_for('register'))
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (username, password, fav_genre, fav_language, age, gender) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (username, generate_password_hash(password, method='pbkdf2:sha256'), fav_genre, fav_language, age, gender))
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except pymysql.err.IntegrityError:
            flash('That username is already taken.', 'danger')
        except Exception as e:
            flash(f'Error: {e}', 'danger')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))



@app.route('/watch_based')
def watch_based():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    if not conn:
        return "Database connection failed", 500

    try:
        cursor = conn.cursor()
        user_id = session['user_id']

        # ==========================================
        # USER INFORMATION
        # ==========================================
        cursor.execute(
            "SELECT id, username, profile_image, gender "
            "FROM users WHERE id=%s",
            (user_id,)
        )
        user = cursor.fetchone()

        if not user:
            session.clear()
            return redirect(url_for('login'))

        if user['profile_image']:
            user['display_image'] = 'uploads/' + user['profile_image']
        elif (user['gender'] or '').lower() == 'female':
            user['display_image'] = 'f_profile.png'
        else:
            user['display_image'] = 'm_profile.png'

        # ==========================================
        # USER PROFILES
        # ==========================================
        cursor.execute(
            "SELECT * FROM user_profiles WHERE user_id=%s",
            (user_id,)
        )
        profiles = cursor.fetchall()

        # ==========================================
        # PERSONALIZED HYBRID RECOMMENDATIONS
        # ==========================================
        deepflix_top = []

        try:
            cursor.execute("""
                SELECT c.id, c.story
                FROM user_history uh
                JOIN contents c ON uh.content_id = c.id
                WHERE uh.user_id = %s
                ORDER BY uh.watched_at DESC
                LIMIT 1
            """, (user_id,))

            last_watched = cursor.fetchone()

            if last_watched:
                deepflix_top = fetch_recommendations(
                    cursor,
                    user_id,
                    last_watched.get('story') or '',
                    current_id=last_watched.get('id'),
                    limit=15
                )

        except Exception as rec_error:
            print(
                "Personalized recommendation error:",
                rec_error
            )
            deepflix_top = []

        # ==========================================
        # FALLBACK: TOP RANKED CONTENT
        # ==========================================
        if not deepflix_top:
            cursor.execute("""
                SELECT *
                FROM contents
                WHERE deepflix_rank > 0
                  AND (is_kids = 0 OR is_kids IS NULL)
                ORDER BY deepflix_rank ASC
                LIMIT 15
            """)
            deepflix_top = cursor.fetchall()

        # ==========================================
        # IMAGE PREPARATION
        # ==========================================
        for movie in deepflix_top:
            movie['display_image'] = (
                movie.get('backdrop_url')
                or movie.get('thumbnail')
            )

        return render_template(
            "watch_based.html",
            user=user,
            profiles=profiles,
            deepflix_top=deepflix_top
        )

    except Exception as e:
        print("Watch based page error:", e)
        return "Unable to load page. Please try again.", 500

    finally:
        conn.close()


@app.route('/select_profile/<profile>')
def select_profile(profile):
    if 'loggedin' not in session or 'user_id' not in session:
        return redirect(url_for('login'))

    if profile == "kids":
        session.update(profile_type="kids", profile_name="Kids", profile_image="kids.png")
        return redirect(url_for('kids_dashboard'))

    if profile == "guest":
        session.update(profile_type="guest", profile_name="Guest", profile_image="m_profile.png")
        return redirect(url_for('user_dashboard'))

    conn = None
    try:
        conn = get_connection()
        if conn is None:
            abort(503, description="Database connection unavailable.")
        cursor = conn.cursor()

        if profile == "main":
            cursor.execute("SELECT username, gender, profile_image FROM users WHERE id=%s LIMIT 1",
                           (session['user_id'],))
            user = cursor.fetchone()
            if not user:
                session.clear()
                return redirect(url_for('login'))
            session['profile_type'] = "main"
            session['profile_name'] = user.get('username') or "User"
            session['profile_image'] = resolve_profile_image(user.get('profile_image'),
                                                             user.get('gender'))
            return redirect(url_for('user_dashboard'))

        try:
            extra_id = int(profile)
        except (TypeError, ValueError):
            return redirect(url_for('watch_based'))

        cursor.execute("SELECT name, gender, profile_image FROM user_profiles "
                       "WHERE id=%s AND user_id=%s LIMIT 1", (extra_id, session['user_id']))
        selected = cursor.fetchone()
        if not selected:
            return redirect(url_for('watch_based'))

        session['profile_type'] = "extra"
        session['profile_name'] = selected.get('name') or "Profile"
        session['profile_image'] = resolve_profile_image(selected.get('profile_image'),
                                                         selected.get('gender'))
        return redirect(url_for('user_dashboard'))

    except Exception as e:
        if hasattr(e, 'code'):  # re-raise abort() errors untouched
            raise
        current_app.logger.exception("Select Profile Error: %s", e)
        abort(500, description="Unable to select profile.")
    finally:
        if conn:
            conn.close()

def verify_password_hash(stored_hash, password):
    try:
        return check_password_hash(stored_hash, password)
    except (ValueError, TypeError):
        app.logger.exception("Password hash verification failed")
        return False
    
@app.route('/add_profile', methods=['GET', 'POST'])
@login_required
def add_profile():
    if request.method == "POST":
        name = request.form.get('name', '').strip()
        gender = request.form.get('gender', 'Other')
        try:
            age = int(request.form.get('age', ''))
        except ValueError:
            flash('Please enter a valid age.', 'danger')
            return redirect(url_for('add_profile'))
        if not name:
            flash('Profile name is required.', 'danger')
            return redirect(url_for('add_profile'))

        conn = get_connection()
        if not conn:
            return "Database connection failed", 500
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM user_profiles WHERE user_id=%s",
                               (session['user_id'],))
                if cursor.fetchone()['total'] >= 2:
                    flash("Maximum 2 profiles allowed", "danger")
                    return redirect(url_for('watch_based'))
                cursor.execute(
                    "INSERT INTO user_profiles (user_id,name,age,gender) VALUES (%s,%s,%s,%s)",
                    (session['user_id'], name, age, gender))
            conn.commit()
        finally:
            conn.close()
        return redirect(url_for('watch_based'))

    return render_template("add_profile.html")


@app.route('/delete_profile/<int:id>')
@login_required
def delete_profile(id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM user_profiles WHERE id=%s AND user_id=%s",
                           (id, session['user_id']))
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for('watch_based'))


@app.route('/edit_profile/<int:id>', methods=['POST'])
@login_required
def edit_profile(id):
    name = request.form.get('name', '').strip()
    try:
        age = int(request.form.get('age', ''))
    except ValueError:
        return redirect(url_for('watch_based'))

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE user_profiles SET name=%s, age=%s WHERE id=%s AND user_id=%s",
                           (name, age, id, session['user_id']))
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for('watch_based'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return "Database connection failed", 500

        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
            user = cursor.fetchone()
            if not user:
                session.clear()
                return redirect(url_for('login'))

            if request.method == 'POST':
                upload_folder = app.config['UPLOAD_FOLDER']
                old_image = user.get('profile_image')

                def remove_old():
                    if old_image:
                        old_path = os.path.join(upload_folder, os.path.basename(old_image))
                        if os.path.isfile(old_path):
                            os.remove(old_path)

                if request.form.get('action') == 'remove_image':
                    remove_old()
                    cursor.execute("UPDATE users SET profile_image=NULL WHERE id=%s", (user_id,))
                    conn.commit()
                    return redirect(url_for('profile'))

                file = request.files.get('profile_image')
                if file and file.filename:
                    try:
                        filename, _ = save_image_file(file, prefix=f"{user_id}_")
                    except ValueError as e:
                        flash(str(e), 'danger')
                        return redirect(url_for('profile'))
                    cursor.execute("UPDATE users SET profile_image=%s WHERE id=%s",
                                   (filename, user_id))
                    conn.commit()
                    if old_image and old_image != filename:
                        remove_old()
                    return redirect(url_for('profile'))

            return render_template('profile.html', user=user)

    except Exception:
        if conn:
            conn.rollback()
        app.logger.exception("Error loading or updating profile")
        return "Unable to load profile. Please try again.", 500
    finally:
        if conn:
            conn.close()


@app.route('/delete_account')
@login_required
def delete_account():
    user_id = session['user_id']
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            for table, column in (
                ('ratings', 'user_id'), ('reviews', 'user_id'), ('likes', 'user_id'),
                ('user_hearts', 'user_id'), ('user_history', 'user_id'),
                ('continue_watching', 'user_id'), ('user_profiles', 'user_id'),
                ('social_graph', 'follower_id'), ('social_graph', 'followed_id'),
                ('chats', 'sender_id'), ('chats', 'receiver_id'),
            ):
                safe_delete(cursor, table, column, user_id)
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        flash(f"Could not delete account: {e}", "danger")
        return redirect(url_for('profile'))
    finally:
        conn.close()

    session.clear()
    flash("Your account has been permanently deleted.", "success")
    return redirect(url_for('register'))


# =========================================================
# ROUTES: KIDS
# =========================================================
@app.route('/kids_dashboard', endpoint='kids_dashboard')
@login_required
def kids_dashboard():
    user_id = session['user_id']
    search_query = request.args.get('search', '')
    language = request.args.get('language', '')
    genre = request.args.get('genre', '')

    conn = get_connection()
    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor()

        query = "SELECT * FROM contents WHERE is_kids = 1"
        params = []
        if search_query:
            query += " AND title LIKE %s"
            params.append(f"%{search_query}%")
        if language:
            query += " AND language = %s"
            params.append(language)
        if genre:
            query += " AND genre = %s"
            params.append(genre)
        query += " ORDER BY release_year DESC"
        cursor.execute(query, params)
        kids_contents = cursor.fetchall()

        cursor.execute("""
            SELECT c.*, cw.progress, cw.duration
            FROM continue_watching cw JOIN contents c ON cw.content_id = c.id
            WHERE cw.user_id = %s AND c.is_kids = 1
            ORDER BY cw.last_watched DESC LIMIT 10
        """, (user_id,))
        continue_watching = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE deepflix_rank > 0 AND is_kids = 1
            ORDER BY deepflix_rank ASC LIMIT 15
        """)
        deepflix_top = cursor.fetchall()

        cursor.execute("""
            SELECT c.*, COUNT(cw.id) AS watch_count
            FROM continue_watching cw JOIN contents c ON cw.content_id = c.id
            WHERE c.is_kids = 1 AND cw.last_watched >= NOW() - INTERVAL 7 DAY
            GROUP BY cw.content_id, c.id
            ORDER BY watch_count DESC LIMIT 7
        """)
        trending_kids = cursor.fetchall()
        if not trending_kids:
            cursor.execute("SELECT * FROM contents WHERE is_kids = 1 "
                           "ORDER BY release_year DESC LIMIT 7")
            trending_kids = cursor.fetchall()

        because_you_watched = []
        cursor.execute("SELECT content_id FROM continue_watching WHERE user_id = %s "
                       "ORDER BY last_watched DESC LIMIT 1", (user_id,))
        last = cursor.fetchone()
        if last:
            cursor.execute("SELECT genre, language FROM contents WHERE id=%s", (last['content_id'],))
            last_content = cursor.fetchone()
            if last_content:
                cursor.execute("""
                    SELECT * FROM contents
                    WHERE is_kids = 1 AND genre = %s AND id != %s LIMIT 7
                """, (last_content['genre'], last['content_id']))
                because_you_watched = cursor.fetchall()

        cursor.execute("SELECT * FROM contents WHERE is_kids = 1 ORDER BY upload_date DESC LIMIT 10")
        new_arrivals = cursor.fetchall()

        cursor.execute("SELECT * FROM contents WHERE is_kids = 1 AND genre LIKE '%%Comedy%%' LIMIT 10")
        funny_kids = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE is_kids = 1
              AND (title LIKE '%%Animal%%' OR title LIKE '%%Dog%%' OR title LIKE '%%Cat%%')
            LIMIT 10
        """)
        animal_kids = cursor.fetchall()

        cursor.execute("SELECT * FROM contents WHERE is_kids = 1 AND language = 'Tamil' LIMIT 10")
        tamil_kids = cursor.fetchall()

        kids_notifications = safe_fetchall(cursor, """
            SELECT n.*, c.title AS content_title
            FROM notifications n LEFT JOIN contents c ON n.content_id = c.id
            WHERE n.user_id = %s AND n.is_kids = 1
            ORDER BY n.created_at DESC LIMIT 10
        """, (user_id,))

        return render_template(
            "kids_dashboard.html",
            kids_contents=kids_contents, continue_watching=continue_watching,
            deepflix_top=deepflix_top, trending_kids=trending_kids,
            because_you_watched=because_you_watched, new_arrivals=new_arrivals,
            funny_kids=funny_kids, animal_kids=animal_kids, tamil_kids=tamil_kids,
            kids_notifications=kids_notifications, search_query=search_query,
            selected_language=language, selected_genre=genre)

    except Exception as e:
        print("KIDS DASHBOARD ERROR:", e)
        return "Something went wrong"
    finally:
        conn.close()


@app.route('/kids_search')
@login_required
def kids_search():
    search_query = request.args.get("q", "").strip()
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()
        like = f"%{search_query}%"
        cursor.execute("""
            SELECT * FROM contents
            WHERE is_kids = 1
              AND (title LIKE %s OR genre LIKE %s OR language LIKE %s
                   OR hero LIKE %s OR heroine LIKE %s OR director LIKE %s)
            ORDER BY release_year DESC
        """, (like,) * 6)
        kids_contents = cursor.fetchall()

        return render_template(
            "kids_dashboard.html",
            kids_contents=kids_contents, continue_watching=[], deepflix_top=[],
            trending=[], trending_kids=[], because_you_watched=[], new_arrivals=[],
            funny_kids=[], animal_kids=[], tamil_kids=[], kids_notifications=[],
            search_query=search_query, selected_language='', selected_genre='')
    except Exception as e:
        print("KIDS SEARCH ERROR:", e)
        return "Search failed"
    finally:
        conn.close()


@app.route('/kids_watch/<int:content_id>')
@login_required
def kids_watch(content_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
            content = cursor.fetchone()
            if not content:
                return "Content not found"
        return render_template("kids_watch.html", content=content)
    finally:
        conn.close()


# =========================================================
# ROUTES: USER DASHBOARD
# =========================================================
@app.route('/user_dashboard', endpoint='user_dashboard')
@login_required
def user_dashboard():
    user_id = session['user_id']
    conn = get_connection()
    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT id, username, profile_image, fav_genre, fav_language "
                       "FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        if not user:
            session.clear()
            return redirect(url_for('login'))

        display_name = session.get('profile_name', user['username'])

        cursor.execute("""
            SELECT * FROM contents
            WHERE backdrop_url IS NOT NULL AND backdrop_url != '' AND is_kids = 0
            ORDER BY deepflix_rank ASC, hearts DESC LIMIT 5
        """)
        hero_banner = cursor.fetchall()

        notifications = safe_fetchall(cursor, """
            SELECT n.*, c.thumbnail AS content_thumbnail
            FROM notifications n LEFT JOIN contents c ON n.content_id = c.id
            WHERE (c.is_kids = 0 OR c.is_kids IS NULL)
            ORDER BY n.created_at DESC LIMIT 10
        """)

        search_query = request.args.get('search', '').strip()
        selected_genre = request.args.get('genre', '').strip()
        selected_language = request.args.get('language', '').strip()

        sql = "SELECT * FROM contents WHERE is_kids = 0"
        params = []
        for word in search_query.split():
            sql += """
                AND (title LIKE %s OR story LIKE %s OR hero LIKE %s OR heroine LIKE %s
                     OR director LIKE %s OR language LIKE %s OR genre LIKE %s)
            """
            params.extend([f"%{word}%"] * 7)
        if selected_genre:
            sql += " AND genre LIKE %s"
            params.append(f"%{selected_genre}%")
        if selected_language:
            sql += " AND language LIKE %s"
            params.append(f"%{selected_language}%")
        sql += " ORDER BY release_year DESC"
        cursor.execute(sql, tuple(params))
        all_contents = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type = 'movie' AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)
            ORDER BY created_at DESC, id DESC LIMIT 15
        """)
        recently_added = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE language LIKE '%%Korean%%' AND is_kids = 0
            ORDER BY hearts DESC LIMIT 15
        """)
        korean_blockbusters = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE deepflix_rank > 0 AND is_kids = 0
            ORDER BY deepflix_rank ASC LIMIT 15
        """)
        deepflix_top = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type='movie' AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL) AND views IS NOT NULL
            ORDER BY views DESC, hearts DESC LIMIT 10
        """)
        trending_movies = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type='series' AND is_kids = 0
            ORDER BY views DESC, hearts DESC LIMIT 10
        """)
        trending_series = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type='movie'
              AND (genre LIKE '%%Crime%%' OR genre LIKE '%%Drama%%') AND is_kids = 0
            ORDER BY hearts DESC LIMIT 10
        """)
        crime_drama_movies = cursor.fetchall()

        cursor.execute("SELECT * FROM contents WHERE is_90s = 1 AND is_kids = 0 "
                       "ORDER BY release_year ASC")
        movies_90s = cursor.fetchall()

        cursor.execute("""
            SELECT c.*, cw.progress
            FROM continue_watching cw JOIN contents c ON cw.content_id=c.id
            WHERE cw.user_id=%s AND c.is_kids = 0
            ORDER BY cw.last_watched DESC LIMIT 10
        """, (user_id,))
        continue_watching = cursor.fetchall()

        # ---------------- Recommendations ----------------
        recommendations = []
        cursor.execute("SELECT COUNT(*) AS total FROM user_history WHERE user_id=%s", (user_id,))
        if cursor.fetchone()['total'] == 0:
            cursor.execute("""
                SELECT * FROM contents
                WHERE genre LIKE %s AND language LIKE %s AND is_kids = 0
                ORDER BY hearts DESC LIMIT 12
            """, (f"%{user.get('fav_genre') or ''}%", f"%{user.get('fav_language') or ''}%"))
            recommendations = cursor.fetchall()
        else:
            cursor.execute("""
                SELECT c.id, c.story
                FROM user_history uh JOIN contents c ON uh.content_id=c.id
                WHERE uh.user_id=%s
                ORDER BY uh.watched_at DESC LIMIT 1
            """, (user_id,))
            last = cursor.fetchone() or {}
            recommendations = fetch_recommendations(
                cursor, user_id, last.get('story') or '', current_id=last.get('id'))

        # PyMySQL fetchall() returns a tuple; normalize recommendations
        # to a list before appending the remaining popular content.
        recommendations = list(recommendations or [])

        if len(recommendations) < 12:
            existing = [r['id'] for r in recommendations]
            fill_sql = "SELECT * FROM contents WHERE is_kids = 0"
            fill_params = []
            if existing:
                fill_sql += f" AND id NOT IN ({','.join(['%s'] * len(existing))})"
                fill_params.extend(existing)
            fill_sql += " ORDER BY hearts DESC LIMIT %s"
            fill_params.append(12 - len(recommendations))
            cursor.execute(fill_sql, tuple(fill_params))
            recommendations.extend(cursor.fetchall())

        return render_template(
            "user_dashboard.html",
            user=user, display_name=display_name, hero_banner=hero_banner,
            notifications=notifications, all_contents=all_contents,
            recently_added=recently_added, korean_blockbusters=korean_blockbusters,
            deepflix_top=deepflix_top, trending_movies=trending_movies,
            trending_series=trending_series, crime_drama_movies=crime_drama_movies,
            movies_90s=movies_90s, continue_watching=continue_watching,
            recommendations=recommendations[:12])

    except Exception as e:
        app.logger.exception("Dashboard Error: %s", e)
        return "Error loading dashboard"
    finally:
        conn.close()


@app.route('/check_notifications')
def check_notifications():
    if 'user_id' not in session:
        return jsonify({"count": 0})
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            rows = safe_fetchall(cursor, """
                SELECT COUNT(*) as count
                FROM notifications n JOIN contents c ON n.content_id = c.id
                WHERE n.user_id = %s AND c.is_kids = 1
            """, (session['user_id'],))
        return jsonify({"count": rows[0]['count'] if rows else 0})
    finally:
        conn.close()


# =========================================================
# ROUTES: CONTENT / PLAYER
# =========================================================
@app.route('/content/<int:content_id>')
@login_required
def content_page(content_id):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
        movie = cursor.fetchone()
        if not movie:
            return "Movie not found"

        touch_continue_watching(cursor, session['user_id'], content_id)
        conn.commit()

        cursor.execute("SELECT * FROM contents WHERE genre LIKE %s AND id != %s LIMIT 10",
                       ("%" + (movie.get('genre') or '') + "%", content_id))
        similar_movies = cursor.fetchall()

        return render_template("content.html", movie=movie, similar_movies=similar_movies)
    except Exception as e:
        print("Content Page Error:", e)
        return "Error loading content page"
    finally:
        conn.close()


CAST_SPEC = [
    ('hero', 'Hero', 'hero_image'), ('hero2', 'Hero', 'hero2_image'),
    ('hero3', 'Hero', 'hero3_image'),
    ('heroine', 'Heroine', 'heroine_image'), ('heroine2', 'Heroine', 'heroine2_image'),
    ('heroine3', 'Heroine', 'heroine3_image'),
    ('comedian', 'Comedian', 'comedian_image'), ('comedian2', 'Comedian', 'comedian2_image'),
    ('director', 'Director', 'director_image'), ('producer', 'Producer', 'producer_image'),
]


def poster_url(item):
    name = (item.get("poster") or "").strip()
    return url_for("static", filename=f"uploads/{name}") if name else None


@app.route('/watch/<int:content_id>')
@login_required
def watch_content(content_id):
    user_id = session['user_id']
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
            content = cursor.fetchone()
            if not content:
                return "Content not found"

            cursor.execute("UPDATE contents SET views = COALESCE(views, 0) + 1 WHERE id=%s",
                           (content_id,))
            content["poster_path"] = poster_url(content)

            cast = []
            for name_key, role, image_key in CAST_SPEC:
                if content.get(name_key):
                    image = content.get(image_key)
                    cast.append({
                        "role": role,
                        "name": content.get(name_key),
                        "image": url_for("static", filename=f"uploads/{image}") if image else None,
                    })

            record_history(cursor, user_id, content_id)
            saved_progress = touch_continue_watching(cursor, user_id, content_id) or 0

            cursor.execute("""
                SELECT r.*, u.username FROM reviews r
                JOIN users u ON r.user_id = u.id
                WHERE r.content_id=%s ORDER BY r.timestamp DESC
            """, (content_id,))
            reviews = cursor.fetchall()

            current_genre = (content.get("genre") or "").strip()
            current_language = (content.get("language") or "").strip()
            current_story = (content.get("story") or "").strip()

            recommended = fetch_recommendations(cursor, user_id, current_story,
                                                current_id=content_id, limit=10)

            if not recommended and current_genre:
                cursor.execute("""
                    SELECT * FROM contents
                    WHERE id != %s AND genre LIKE %s AND language LIKE %s
                    ORDER BY hearts DESC LIMIT 12
                """, (content_id, f"%{current_genre}%", f"%{current_language}%"))
                recommended = cursor.fetchall()

            if not recommended and current_genre:
                cursor.execute("""
                    SELECT * FROM contents
                    WHERE id != %s AND genre LIKE %s
                    ORDER BY RAND() LIMIT 12
                """, (content_id, f"%{current_genre}%"))
                recommended = cursor.fetchall()

            for item in recommended:
                item["poster_path"] = poster_url(item)

        conn.commit()
        return render_template("watch.html", content=content, cast=cast, reviews=reviews,
                               saved_progress=saved_progress, recommended=recommended)
    except Exception as e:
        app.logger.exception("WATCH PAGE ERROR: %s", e)
        return "Something went wrong"
    finally:
        conn.close()


@app.route('/play/<int:content_id>')
@login_required
def play_content(content_id):
    user_id = session['user_id']
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
        movie = cursor.fetchone()
        if not movie:
            return "Content not found"

        record_history(cursor, user_id, content_id)
        touch_continue_watching(cursor, user_id, content_id)
        conn.commit()

        recommendations = fetch_recommendations(cursor, user_id, movie.get('story') or '',
                                                current_id=content_id, limit=10)
        return render_template("player.html", movie=movie, recommendations=recommendations)
    except Exception as e:
        print("Player Error:", e)
        return "Error loading player"
    finally:
        conn.close()


@app.route('/all_movies')
@login_required
def all_movies():
    search = request.args.get('search', '').strip()
    genre = request.args.get('genre', '').strip()

    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM contents WHERE (is_kids = 0 OR is_kids IS NULL)"
        params = []
        if search:
            pattern = "%" + search.lower() + "%"
            query += """
                AND (LOWER(title) LIKE %s OR LOWER(hero) LIKE %s OR LOWER(heroine) LIKE %s
                     OR LOWER(director) LIKE %s OR LOWER(language) LIKE %s OR LOWER(genre) LIKE %s)
            """
            params.extend([pattern] * 6)
        if genre and genre.lower() != "all":
            query += " AND LOWER(genre) LIKE %s"
            params.append("%" + genre.lower() + "%")
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        movies = cursor.fetchall()

        cursor.execute("SELECT username, profile_image FROM users WHERE id=%s", (session['user_id'],))
        user = cursor.fetchone()

        return render_template("all_movies.html", movies=movies, user=user, search_value=search)
    finally:
        conn.close()


@app.route('/movie_suggestions')
def movie_suggestions():
    keyword = request.args.get('q', '').strip()
    if not keyword:
        return jsonify([])

    conn = get_connection()
    try:
        cursor = conn.cursor()
        pattern = "%" + keyword.lower() + "%"
        cursor.execute("""
            SELECT title, hero, heroine, director, genre, language
            FROM contents
            WHERE content_type='movie'
              AND (LOWER(title) LIKE %s OR LOWER(hero) LIKE %s OR LOWER(heroine) LIKE %s
                   OR LOWER(director) LIKE %s OR LOWER(genre) LIKE %s OR LOWER(language) LIKE %s)
            LIMIT 25
        """, (pattern,) * 6)
        suggestions = set()
        for row in cursor.fetchall():
            for field in ('title', 'hero', 'heroine', 'director', 'genre', 'language'):
                if row.get(field):
                    suggestions.add(row[field])

        ranked = sorted(suggestions,
                        key=lambda x: fuzz.partial_ratio(keyword.lower(), x.lower()),
                        reverse=True)
        return jsonify(ranked[:10])
    finally:
        conn.close()


@app.route('/person/<name>')
def person_movies(name):
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return "Database connection failed", 500
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM contents
                WHERE hero = %s OR hero2 = %s OR hero3 = %s
                   OR heroine = %s OR heroine2 = %s OR heroine3 = %s
                   OR director = %s OR producer = %s OR comedian = %s
            """, (name,) * 9)
            movies = cursor.fetchall()

        person = {"name": name, "profile_image": name.lower().replace(" ", "_") + ".jpg"}
        return render_template("person_movies.html", name=name, movies=movies, person=person)
    except Exception:
        app.logger.exception("Error loading movies for person: %s", name)
        return "Unable to load movies right now. Please try again.", 500
    finally:
        if conn:
            conn.close()


@app.route('/all_series')
@login_required
def all_series():
    user_id = session['user_id']
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT n.*, c.thumbnail AS content_thumbnail
            FROM notifications n JOIN contents c ON n.content_id = c.id
            WHERE c.content_type = 'series' AND c.is_kids = 0
            ORDER BY n.created_at DESC LIMIT 10
        """)
        notifications = cursor.fetchall()

        cursor.execute("""
            SELECT c.*, cw.progress
            FROM continue_watching cw JOIN contents c ON cw.content_id = c.id
            WHERE cw.user_id = %s AND c.content_type = 'series' AND c.is_kids = 0
            ORDER BY cw.last_watched DESC LIMIT 10
        """, (user_id,))
        continue_watching_series = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type = 'series' AND is_kids = 0 AND created_at IS NOT NULL
            ORDER BY created_at DESC, id DESC LIMIT 15
        """)
        recently_added_series = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type = 'series' AND is_kids = 0
            ORDER BY release_year DESC
        """)
        series_list = cursor.fetchall()

        return render_template('all_series.html', notifications=notifications,
                               continue_watching_series=continue_watching_series,
                               recently_added_series=recently_added_series,
                               series_list=series_list)
    except Exception as e:
        return str(e)
    finally:
        conn.close()


@app.route('/series_play/<int:content_id>')
@login_required
def series_play(content_id):
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contents WHERE id = %s AND content_type = 'series'",
                       (content_id,))
        series = cursor.fetchone()
        if not series:
            return "Series not found"

        full_title = series['title'] or ''
        base_title = full_title.split(":")[0].strip() if ":" in full_title else full_title.strip()

        cursor.execute("""
            SELECT * FROM contents
            WHERE content_type = 'series' AND title LIKE %s AND is_kids = 0
            ORDER BY id ASC
        """, (f"{base_title}:%",))
        episodes = cursor.fetchall()

        return render_template('series_play.html', series=series, episodes=episodes,
                               base_title=base_title)
    except Exception as e:
        return str(e)
    finally:
        conn.close()


# =========================================================
# ROUTES: SOCIAL
# =========================================================
def add_initial(row):
    row['initial'] = (row.get('username') or '?')[0].upper()
    return row


@app.route('/social_hub')
@login_required
def social_hub():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, gender, profile_image FROM users WHERE id = %s",
                           (session['user_id'],))
            current_user = cursor.fetchone()
            if not current_user:
                session.clear()
                return redirect(url_for('login'))
            add_initial(current_user)

            cursor.execute("SELECT id, username, age, fav_genre, gender, profile_image "
                           "FROM users WHERE id != %s", (session['user_id'],))
            users = [add_initial(u) for u in cursor.fetchall()]

            cursor.execute("SELECT followed_id FROM social_graph WHERE follower_id=%s",
                           (session['user_id'],))
            following = [row['followed_id'] for row in cursor.fetchall()]

            return render_template('social_hub.html', users=users, following=following,
                                   current_user=current_user)
    except Exception as e:
        print("SOCIAL HUB ERROR:", e)
        return "Something went wrong", 500
    finally:
        conn.close()


@app.route('/socialhub_profile/<int:user_id>')
@login_required
def socialhub_profile(user_id):
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT id, username, gender, profile_image FROM users WHERE id=%s",
                       (session['user_id'],))
        current_user = cursor.fetchone()
        if not current_user:
            session.clear()
            return redirect(url_for('login'))
        add_initial(current_user)

        cursor.execute("SELECT id, username, age, fav_genre, gender, profile_image "
                       "FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return "User not found", 404
        add_initial(user)

        cursor.execute("""
            SELECT c.id, c.title, c.genre, c.thumbnail, MAX(uh.watched_at) AS last_seen
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id=%s
            GROUP BY c.id, c.title, c.genre, c.thumbnail
            ORDER BY last_seen DESC LIMIT 5
        """, (user_id,))
        history_based = cursor.fetchall()

        cursor.execute("""
            SELECT id, title, genre, thumbnail FROM contents
            WHERE genre LIKE %s ORDER BY trending_rank DESC LIMIT 5
        """, (f"%{user.get('fav_genre') or ''}%",))
        genre_based = cursor.fetchall()

        combined = {item['id']: item for item in history_based}
        for item in genre_based:
            combined.setdefault(item['id'], item)
        recommended_contents = list(combined.values())

        for item in recommended_contents:
            thumb = item.get('thumbnail')
            if thumb and not thumb.startswith(('http://', 'https://')):
                item['thumbnail'] = url_for('static', filename=f'uploads/{thumb}')

        cursor.execute("SELECT 1 FROM social_graph WHERE follower_id=%s AND followed_id=%s",
                       (session['user_id'], user_id))
        is_following = cursor.fetchone() is not None

        cursor.execute("""
            SELECT u.id, u.username, u.profile_image, u.gender
            FROM social_graph sg JOIN users u ON sg.follower_id = u.id
            WHERE sg.followed_id=%s
        """, (user_id,))
        followers = [add_initial(f) for f in cursor.fetchall()]

        cursor.execute("""
            SELECT u.id, u.username, u.profile_image, u.gender
            FROM social_graph sg JOIN users u ON sg.followed_id = u.id
            WHERE sg.follower_id=%s
        """, (user_id,))
        following = [add_initial(f) for f in cursor.fetchall()]

        return render_template('socialhub_profile.html', user=user,
                               recommended_contents=recommended_contents,
                               is_following=is_following, followers=followers,
                               following=following, current_user=current_user)
    except Exception as e:
        print("PROFILE ERROR:", e)
        return "Something went wrong", 500
    finally:
        conn.close()


@app.route('/toggle_follow/<int:user_id>', methods=['POST'])
def toggle_follow(user_id):
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"})

    me = session['user_id']
    if me == user_id:
        return jsonify({"success": False, "message": "You cannot follow yourself"})

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM social_graph WHERE follower_id=%s AND followed_id=%s",
                           (me, user_id))
            if cursor.fetchone():
                cursor.execute("DELETE FROM social_graph WHERE follower_id=%s AND followed_id=%s",
                               (me, user_id))
                following_status = False
            else:
                cursor.execute("INSERT INTO social_graph (follower_id, followed_id) VALUES (%s,%s)",
                               (me, user_id))
                following_status = True
            conn.commit()
        return jsonify({"success": True, "following": following_status, "user_id": user_id})
    finally:
        conn.close()


@app.route('/api/follow/<int:target_id>')
@login_required
def api_follow(target_id):
    if target_id == session['user_id']:
        return redirect(url_for('social_hub'))
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM social_graph WHERE follower_id=%s AND followed_id=%s",
                           (session['user_id'], target_id))
            exists = cursor.fetchone()
            if exists:
                cursor.execute("DELETE FROM social_graph WHERE id=%s", (exists['id'],))
                flash('Unfollowed user.', 'info')
            else:
                cursor.execute("INSERT INTO social_graph (follower_id, followed_id) VALUES (%s, %s)",
                               (session['user_id'], target_id))
                flash('Followed user!', 'success')
        conn.commit()
        return redirect(url_for('social_hub'))
    finally:
        conn.close()


def format_chat_user(user):
    if user.get('profile_image'):
        user['profile_image'] = '/static/uploads/' + user['profile_image']
        user['initial'] = None
    else:
        user['profile_image'] = None
        user['initial'] = user['username'][0].upper()
    return user


@app.route('/chat/<int:user_id>')
@login_required
def chat(user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, profile_image FROM users WHERE id=%s", (user_id,))
            chat_user = cursor.fetchone()
            if not chat_user:
                return "User not found", 404
            cursor.execute("SELECT id, username, profile_image FROM users WHERE id=%s",
                           (session['user_id'],))
            me = cursor.fetchone()
            return render_template('chat.html', chat_user=format_chat_user(chat_user),
                                   me=format_chat_user(me))
    finally:
        conn.close()


@app.route('/fetch_messages/<int:user_id>')
def fetch_messages(user_id):
    if 'user_id' not in session:
        return jsonify([])

    me_id = session['user_id']
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT sender_id, receiver_id, message, timestamp
                FROM chats
                WHERE (sender_id=%s AND receiver_id=%s) OR (sender_id=%s AND receiver_id=%s)
                ORDER BY timestamp ASC
            """, (me_id, user_id, user_id, me_id))
            messages = cursor.fetchall()

            sender_ids = list({m['sender_id'] for m in messages})
            users = {}
            if sender_ids:
                ph = ','.join(['%s'] * len(sender_ids))
                cursor.execute(f"SELECT id, username, profile_image FROM users WHERE id IN ({ph})",
                               tuple(sender_ids))
                users = {u['id']: u for u in cursor.fetchall()}

            for msg in messages:
                sender = users.get(msg['sender_id'])
                if sender:
                    msg['username'] = sender['username']
                    if sender.get('profile_image'):
                        msg['profile_image'] = '/static/uploads/' + sender['profile_image']
                        msg['initial'] = None
                    else:
                        msg['profile_image'] = None
                        msg['initial'] = sender['username'][0].upper()
                else:
                    msg['username'], msg['profile_image'], msg['initial'] = "Unknown", None, "?"
                msg['is_me'] = (msg['sender_id'] == me_id)

            return jsonify(messages)
    finally:
        conn.close()


@app.route('/send_message/<int:user_id>', methods=['POST'])
def send_message(user_id):
    if 'user_id' not in session:
        return jsonify({"success": False})

    msg = ((request.get_json(silent=True) or {}).get('message') or '').strip()
    if not msg:
        return jsonify({"success": False})

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO chats (sender_id, receiver_id, message) VALUES (%s,%s,%s)",
                           (session['user_id'], user_id, msg))
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()


# =========================================================
# ROUTES: RATINGS / REVIEWS / PROGRESS / HEARTS (JSON APIs)
# =========================================================
@app.route('/api/rate', methods=['POST'])
def api_rate():
    if 'user_id' not in session:
        return jsonify({'error': 'Login required'}), 401

    data = request.get_json(silent=True) or {}
    content_id = data.get('content_id')
    try:
        rating = float(data.get('rating'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Missing data'}), 400
    if not content_id or not (0 < rating <= 5):
        return jsonify({'error': 'Invalid data'}), 400

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM ratings WHERE user_id=%s AND content_id=%s",
                           (session['user_id'], content_id))
            if cursor.fetchone():
                cursor.execute("UPDATE ratings SET rating=%s WHERE user_id=%s AND content_id=%s",
                               (rating, session['user_id'], content_id))
            else:
                cursor.execute("INSERT INTO ratings (user_id, content_id, rating) VALUES (%s, %s, %s)",
                               (session['user_id'], content_id, rating))
        conn.commit()
        ALSRecommender._cache.pop(session['user_id'], None)  # refresh recommendations
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/rating/get')
def get_rating():
    if 'user_id' not in session:
        return jsonify({'success': False, 'rating': 0})

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT rating FROM ratings WHERE user_id=%s AND content_id=%s",
                           (session['user_id'], request.args.get('content_id')))
            result = cursor.fetchone()
            return jsonify({'success': True, 'rating': result['rating'] if result else 0})
    finally:
        conn.close()


@app.route('/api/review', methods=['POST'])
@login_required
def api_review():
    content_id = request.form.get('content_id', type=int)
    text = request.form.get('review_text', '').strip()
    if not content_id or not text:
        flash('Review text is required.', 'danger')
        return redirect(request.referrer or url_for('user_dashboard'))

    sentiment = TextBlob(text).sentiment.polarity  # -1 .. 1

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO reviews (user_id, content_id, review_text, sentiment_score) "
                "VALUES (%s, %s, %s, %s)", (session['user_id'], content_id, text, sentiment))
        conn.commit()
        flash('Review posted!', 'success')
        return redirect(url_for('watch_content', content_id=content_id))
    finally:
        conn.close()


@app.route('/api/progress/get')
def get_progress():
    if 'user_id' not in session:
        return jsonify({'success': False, 'progress': 0})
    content_id = request.args.get('content_id')
    if not content_id:
        return jsonify({'success': False, 'progress': 0})

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT progress FROM continue_watching WHERE user_id=%s AND content_id=%s",
                           (session['user_id'], content_id))
            row = cursor.fetchone()
        return jsonify({'success': True, 'progress': float(row['progress']) if row else 0})
    finally:
        conn.close()


@app.route('/api/progress/update', methods=['POST'])
def update_progress():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401

    data = request.get_json(silent=True) or {}
    content_id = data.get('content_id')
    try:
        progress = float(data.get('progress'))
    except (TypeError, ValueError):
        return jsonify({'success': False}), 400
    if not content_id:
        return jsonify({'success': False}), 400

    progress = max(0, min(progress, 100))

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if progress >= 95:   # finished -> remove from "continue watching"
                cursor.execute("DELETE FROM continue_watching WHERE user_id=%s AND content_id=%s",
                               (session['user_id'], content_id))
            else:
                touch_continue_watching(cursor, session['user_id'], content_id, progress)
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        print("Progress Error:", e)
        return jsonify({'success': False}), 500
    finally:
        conn.close()


@app.route('/ajax_toggle_heart', methods=['POST'])
def ajax_toggle_heart():
    if 'user_id' not in session:
        return jsonify({'status': 'login_required'}), 401

    content_id = (request.get_json(silent=True) or {}).get('content_id')
    if not content_id:
        return jsonify({'status': 'error', 'message': 'Content ID is required'}), 400

    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM contents WHERE id=%s", (content_id,))
            if not cursor.fetchone():
                return jsonify({'status': 'error', 'message': 'Content not found'}), 404

            cursor.execute("SELECT 1 FROM user_hearts WHERE user_id=%s AND content_id=%s",
                           (session['user_id'], content_id))
            if cursor.fetchone():
                cursor.execute("DELETE FROM user_hearts WHERE user_id=%s AND content_id=%s",
                               (session['user_id'], content_id))
                cursor.execute("UPDATE contents SET hearts = GREATEST(COALESCE(hearts, 0) - 1, 0) "
                               "WHERE id=%s", (content_id,))
            else:
                cursor.execute("INSERT INTO user_hearts (user_id, content_id) VALUES (%s, %s)",
                               (session['user_id'], content_id))
                cursor.execute("UPDATE contents SET hearts = COALESCE(hearts, 0) + 1 WHERE id=%s",
                               (content_id,))
            conn.commit()

            cursor.execute("SELECT hearts FROM contents WHERE id=%s", (content_id,))
            updated = cursor.fetchone()

        return jsonify({'status': 'success', 'hearts': updated['hearts'] if updated else 0})
    except Exception:
        if conn:
            conn.rollback()
        app.logger.exception("Error toggling content heart")
        return jsonify({'status': 'error',
                        'message': 'Unable to update heart. Please try again.'}), 500
    finally:
        if conn:
            conn.close()


# =========================================================
# ROUTES: ADMIN ANALYTICS
# =========================================================
@app.route('/admin_dashboard_analysis')
@admin_required
def admin_dashboard_analysis():
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()

        def rows(sql):
            cursor.execute(sql)
            return cursor.fetchall() or []

        def with_images(items, key):
            for item in items:
                item['image'] = (item.get(key) or '').lower().replace(" ", "_") + ".jpg"
            return items

        cursor.execute("SELECT COUNT(*) AS total_movies FROM contents")
        total_movies = (cursor.fetchone() or {}).get('total_movies', 0)

        languages = rows("SELECT language, COUNT(*) AS count FROM contents "
                         "GROUP BY language ORDER BY count DESC")
        genres = rows("SELECT genre, COUNT(*) AS count FROM contents "
                      "GROUP BY genre ORDER BY count DESC")
        heroes = with_images(rows(
            "SELECT hero, COUNT(*) AS count FROM contents "
            "WHERE hero IS NOT NULL AND hero != '' GROUP BY hero ORDER BY count DESC"), 'hero')
        heroines = with_images(rows(
            "SELECT heroine, COUNT(*) AS count FROM contents "
            "WHERE heroine IS NOT NULL AND heroine != '' GROUP BY heroine ORDER BY count DESC"),
            'heroine')
        directors = with_images(rows(
            "SELECT director, COUNT(*) AS count FROM contents "
            "WHERE director IS NOT NULL AND director != '' GROUP BY director ORDER BY count DESC"),
            'director')
        producers = with_images(rows(
            "SELECT producer, COUNT(*) AS count FROM contents "
            "WHERE producer IS NOT NULL AND producer != '' GROUP BY producer ORDER BY count DESC"),
            'producer')
        most_watched = rows("""
            SELECT c.id, c.title, c.thumbnail, COUNT(*) AS views
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            GROUP BY c.id, c.title, c.thumbnail ORDER BY views DESC""")
        trending = rows("SELECT id, title, thumbnail, hearts FROM contents ORDER BY hearts DESC")
        timeline = rows("SELECT DATE(watched_at) AS day, COUNT(*) AS views FROM user_history "
                        "GROUP BY DATE(watched_at) ORDER BY day ASC")
        users = rows("SELECT id, username, profile_image FROM users ORDER BY id DESC")
        top_users = rows("""
            SELECT u.id, u.username, COUNT(uh.id) AS count
            FROM users u LEFT JOIN user_history uh ON u.id = uh.user_id
            GROUP BY u.id, u.username ORDER BY count DESC""")
        posters = rows("""
            SELECT id, title, thumbnail, genre, language, release_year, hearts
            FROM contents ORDER BY id DESC LIMIT 20""")

        confusion_matrix, labels = [], []
        accuracy, val_accuracy, loss, val_loss = [], [], [], []
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r") as f:
                    metrics = json.load(f)
                confusion_matrix = metrics.get("confusion_matrix") or []
                labels = metrics.get("labels") or []
                accuracy = metrics.get("accuracy") or []
                val_accuracy = metrics.get("val_accuracy") or []
                loss = metrics.get("loss") or []
                val_loss = metrics.get("val_loss") or []
            except Exception as e:
                print("Metrics Load Error:", e)
        epochs = list(range(1, len(accuracy) + 1))

        return render_template(
            "admin_dashboard_analysis.html",
            total_movies=total_movies, languages=languages, genres=genres,
            heroes=heroes, heroines=heroines, directors=directors, producers=producers,
            most_watched=most_watched, trending=trending, timeline=timeline,
            users=users, top_users=top_users, posters=posters,
            confusion_matrix=confusion_matrix, labels=labels, epochs=epochs,
            accuracy=accuracy, val_accuracy=val_accuracy, loss=loss, val_loss=val_loss)
    except Exception as e:
        print("Admin Dashboard Error:", e)
        return "Error loading dashboard"
    finally:
        conn.close()


@app.route('/api/model-metrics')
@admin_required
def get_model_metrics():
    try:
        with open(METRICS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})


def render_person_analysis(role, name):
    """Shared by producer / hero / heroine / director analysis pages.
    `role` is always a hard-coded column name."""
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM contents WHERE {role} = %s ORDER BY release_year DESC",
                       (name,))
        movies = cursor.fetchall() or []
        cursor.execute(f"SELECT genre, COUNT(*) AS count FROM contents "
                       f"WHERE {role} = %s GROUP BY genre", (name,))
        genres = cursor.fetchall() or []
        cursor.execute(f"SELECT language, COUNT(*) AS count FROM contents "
                       f"WHERE {role} = %s GROUP BY language", (name,))
        languages = cursor.fetchall() or []

        return render_template(
            f"admin_{role}_analysis.html",
            **{role: name, f"{role}_image": name.lower().replace(" ", "_") + ".jpg"},
            movies=movies, genres=genres, languages=languages)
    except Exception as e:
        print(f"{role} analysis error:", e)
        return f"{role.capitalize()} analysis error"
    finally:
        conn.close()


@app.route('/admin_producer_analysis/<producer_name>')
@admin_required
def admin_producer_analysis(producer_name):
    return render_person_analysis('producer', producer_name)


@app.route('/admin_hero_analysis/<hero>')
@admin_required
def admin_hero_analysis(hero):
    return render_person_analysis('hero', hero)


@app.route('/admin_heroine_analysis/<heroine>')
@admin_required
def admin_heroine_analysis(heroine):
    return render_person_analysis('heroine', heroine)


@app.route('/admin_director_analysis/<director>')
@admin_required
def admin_director_analysis(director):
    return render_person_analysis('director', director)


@app.route('/admin_user_analysis/<int:user_id>')
@admin_required
def admin_user_analysis(user_id):
    conn = get_connection()
    if not conn:
        return "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, profile_image FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return "User not found", 404

        cursor.execute("SELECT COUNT(*) AS total_views FROM user_history WHERE user_id = %s",
                       (user_id,))
        user['total_views'] = cursor.fetchone()['total_views']

        cursor.execute("""
            SELECT c.id, c.title, c.genre, c.language, c.thumbnail, COUNT(*) AS views
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.id, c.title, c.genre, c.language, c.thumbnail
            ORDER BY views DESC LIMIT 12
        """, (user_id,))
        movies = cursor.fetchall() or []

        cursor.execute("""
            SELECT c.genre, COUNT(*) AS count
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s GROUP BY c.genre
        """, (user_id,))
        genres = cursor.fetchall() or []

        cursor.execute("""
            SELECT c.language, COUNT(*) AS count
            FROM user_history uh JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s GROUP BY c.language
        """, (user_id,))
        languages = cursor.fetchall() or []

        return render_template("admin_user_analysis.html", user=user, movies=movies,
                               genres=genres, languages=languages)
    except Exception as e:
        print("Admin user analysis error:", e)
        return "Error loading user analysis"
    finally:
        conn.close()


def json_query(sql):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        conn.close()


@app.route('/api/user-growth')
@admin_required
def user_growth():
    return jsonify(json_query("""
        SELECT DATE(created_at) as day, COUNT(*) as users
        FROM users GROUP BY DATE(created_at) ORDER BY day ASC
    """))


@app.route('/api/genre-trend')
@admin_required
def genre_trend():
    return jsonify(json_query("""
        SELECT genre, DATE(created_at) as day, COUNT(*) as count
        FROM contents GROUP BY genre, DATE(created_at) ORDER BY day ASC
    """))


@app.route('/api/movie-success')
@admin_required
def movie_success():
    return jsonify(json_query("""
        SELECT c.title,
               COUNT(DISTINCT h.user_id) AS views,
               COUNT(DISTINCT uh.user_id) AS hearts,
               (COUNT(DISTINCT h.user_id) * 0.7 + COUNT(DISTINCT uh.user_id) * 1.3) AS score
        FROM contents c
        LEFT JOIN user_history h ON c.id = h.content_id
        LEFT JOIN user_hearts uh ON c.id = uh.content_id
        GROUP BY c.id, c.title
        ORDER BY score DESC LIMIT 10
    """))


@app.route('/api/user-retention')
@admin_required
def user_retention():
    data = json_query("""
        SELECT DATE(created_at) as day, COUNT(*) as new_users
        FROM users GROUP BY DATE(created_at) ORDER BY day ASC
    """)
    # NOTE: returning_users is an estimate (60% of new users), not real tracking data.
    return jsonify([{"day": r["day"], "new_users": r["new_users"],
                     "returning_users": int(r["new_users"] * 0.6)} for r in data])


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1", use_reloader=False)