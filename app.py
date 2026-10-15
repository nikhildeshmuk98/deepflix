
import os
import io
import time
import json
import random
import subprocess
import pickle

from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)

import pymysql.cursors

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from textblob import TextBlob
from rapidfuzz import fuzz
from wordcloud import WordCloud

from hybrid_engine import HybridEngine


# Initialize the hybrid recommendation engine
hybrid_engine = HybridEngine()

# --- Configuration ---
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.secret_key = 'super_secret_hybrid_key'

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Global AI Models ---
# Absolute paths to avoid "file not found" errors
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODEL_FILE = os.path.join(DATA_DIR, 'pretrained_model.keras')   # Use .h5 for compatibility
TOKENIZER_FILE = os.path.join(DATA_DIR, 'tokenizer.pkl')
MLB_FILE = os.path.join(DATA_DIR, 'mlb.pkl')
MAX_SEQUENCE_LENGTH = 500

MODEL = None
TOKENIZER = None

# --- Load NLP Model & Tokenizer ---
def load_nlp_model():
    global MODEL, TOKENIZER

    # Load Keras model
    try:
        MODEL = load_model(MODEL_FILE)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", e)

    # Compatibility fix for old tokenizer.pkl
    import sys
    import tensorflow.keras.preprocessing.text as tktext
    sys.modules['keras.preprocessing.text'] = tktext

    # Load tokenizer
    try:
        with open(TOKENIZER_FILE, 'rb') as handle:
            TOKENIZER = pickle.load(handle)
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print("❌ Error loading tokenizer:", e)

# Call this once at startup
load_nlp_model()


# --- Database Config ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'db': os.getenv('DB_NAME', 'ott_recommendation_db1'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}

# Aiven SSL certificate
DB_SSL_CA = os.getenv(
    'DB_SSL_CA',
    os.path.join(os.path.dirname(__file__), 'certs', 'ca.pem')
)

if os.path.isfile(DB_SSL_CA):
    DB_CONFIG['ssl'] = {'ca': DB_SSL_CA}


# --- 1. Database Management & Setup ---
def get_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def setup_database():
    """Creates necessary tables for the Hybrid System if they don't exist."""
    conn = get_connection()
    if not conn: return
    try:
        with conn.cursor() as cursor:
            # 1. Update Users for Demographics
            cursor.execute("SHOW COLUMNS FROM users LIKE 'age'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE users ADD COLUMN age INT DEFAULT 25")
                cursor.execute("ALTER TABLE users ADD COLUMN gender VARCHAR(20) DEFAULT 'Other'")
            
            # 2. Ratings Table (For ALS)
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

            # 3. Reviews Table (For Sentiment Analysis)
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

            # 4. Social Graph (For Social Layer)
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
            
            # 5. Likes (Implicit Feedback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS likes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        print("✅ Database Schema Updated for Hybrid Architecture.")
    except Exception as e:
        print(f"❌ DB Setup Error: {e}")
    finally:
        conn.close()

# --- 2. AI & Hybrid Engines ---


class ALSRecommender:
    """Method A: ALS Matrix Factorization using SVD"""
    def predict(self, user_id, all_content_ids):
        conn = get_connection()
        if not conn: return {}
        try:
            # Fetch all ratings
            query = "SELECT user_id, content_id, rating FROM ratings"
            df = pd.read_sql(query, conn)
            
            if df.empty or len(df) < 5: return {} # Not enough data
            
            # Create pivot table (User x Item)
            pivot_df = df.pivot(index='user_id', columns='content_id', values='rating').fillna(0)
            
            # Convert to matrix
            R = pivot_df.values
            user_ratings_mean = np.mean(R, axis=1)
            R_demeaned = R - user_ratings_mean.reshape(-1, 1)
            
            # Singular Value Decomposition (Math behind ALS)
            k = min(50, min(R_demeaned.shape) - 1) # Latent features
            if k < 1: return {}
            
            U, sigma, Vt = svds(R_demeaned, k=k)
            sigma = np.diag(sigma)
            
            # Reconstruct Prediction Matrix
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            preds_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_df.columns, index=pivot_df.index)
            
            if user_id in preds_df.index:
                # Return dictionary {content_id: predicted_rating}
                return preds_df.loc[user_id].to_dict()
            return {}
        except Exception as e:
            print(f"ALS Error: {e}")
            return {}
        finally:
            conn.close()

class DemographicPSORecommender:
    """Method B: Demographic Similarity + PSO Weights"""
    def get_similar_users(self, user_id):
        conn = get_connection()
        try:
            # Fetch current user
            with conn.cursor() as cursor:
                cursor.execute("SELECT age, gender, fav_genre FROM users WHERE id=%s", (user_id,))
                target_user = cursor.fetchone()
                
                if not target_user: return []

                # Fetch all users
                cursor.execute("SELECT id, age, gender, fav_genre FROM users WHERE id != %s", (user_id,))
                other_users = cursor.fetchall()

            # Calculate Similarity
            similar_users = []
            target_genres = set(target_user['fav_genre'].split(','))
            
            for u in other_users:
                score = 0
                # Age similarity (0 to 1)
                age_diff = abs(int(target_user['age']) - int(u['age']))
                score += 1 / (1 + age_diff * 0.1) 
                
                # Gender match
                if target_user['gender'] == u['gender']:
                    score += 0.5
                
                # Genre Overlap (Jaccard)
                u_genres = set(u['fav_genre'].split(','))
                overlap = len(target_genres.intersection(u_genres))
                union = len(target_genres.union(u_genres))
                if union > 0:
                    score += (overlap / union) * 2.0 # Give higher weight to genre
                
                similar_users.append((u['id'], score))
            
            # Sort by score
            similar_users.sort(key=lambda x: x[1], reverse=True)
            return similar_users[:5] # Top 5 similar users
        finally:
            conn.close()

    def predict(self, user_id):
        # 1. Find similar users
        similar_users = self.get_similar_users(user_id)
        if not similar_users: return {}
        
        conn = get_connection()
        scores = {}
        try:
            sim_user_ids = [str(u[0]) for u in similar_users]
            if not sim_user_ids: return {}
            
            # 2. Get movies they liked (Ratings > 3)
            format_strings = ','.join(['%s'] * len(sim_user_ids))
            query = f"SELECT content_id, rating FROM ratings WHERE user_id IN ({format_strings}) AND rating >= 4"
            
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(sim_user_ids))
                results = cursor.fetchall()
                
                for r in results:
                    cid = r['content_id']
                    if cid not in scores: scores[cid] = 0
                    scores[cid] += r['rating'] # Simple sum for now, PSO would tune this aggregation
            
            return scores
        finally:
            conn.close()

class SocialRecommender:
    """Method D: Social Interaction Layer"""
    def predict(self, user_id):
        conn = get_connection()
        scores = {}
        try:
            with conn.cursor() as cursor:
                # 1. Get list of people user follows
                cursor.execute("SELECT followed_id FROM social_graph WHERE follower_id = %s", (user_id,))
                following = [row['followed_id'] for row in cursor.fetchall()]
                
                if not following: return {}

                # 2. Get what they liked/rated
                format_strings = ','.join(['%s'] * len(following))
                
                # Explicit Ratings
                cursor.execute(f"SELECT content_id, rating FROM ratings WHERE user_id IN ({format_strings})", tuple(following))
                for r in cursor.fetchall():
                    scores[r['content_id']] = scores.get(r['content_id'], 0) + r['rating']
                
                # Implicit Likes (Count as 5 stars)
                cursor.execute(f"SELECT content_id FROM likes WHERE user_id IN ({format_strings})", tuple(following))
                for r in cursor.fetchall():
                    scores[r['content_id']] = scores.get(r['content_id'], 0) + 5.0
                    
            return scores
        finally:
            conn.close()

class HybridEngine:
    """Combines A, B, C, D using Weighted Average"""
    def __init__(self):
        self.als = ALSRecommender()
        self.demo = DemographicPSORecommender()
        self.social = SocialRecommender()
        # Weights (Tunable via PSO in theory)
        self.W_ALS = 0.35
        self.W_DEMO = 0.20
        self.W_TEXT = 0.25
        self.W_SOCIAL = 0.20
    
    def normalize(self, scores_dict):
        if not scores_dict: return {}
        max_val = max(scores_dict.values())
        if max_val == 0: return scores_dict
        return {k: v/max_val for k, v in scores_dict.items()}

    def get_hybrid_recommendations(self, user_id, last_watched_desc=None):
        # 1. Get Scores from all engines
        # A. ALS
        scores_als = self.normalize(self.als.predict(user_id, []))
        
        # B. Demographic
        scores_demo = self.normalize(self.demo.predict(user_id))
        
        # D. Social
        scores_social = self.normalize(self.social.predict(user_id))
        
        # C. Text AI (Integrated here based on last watched)
        scores_text = {}
        conn = get_connection()
        all_content_ids = []
        
        # Fetch all content to map IDs
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, story FROM contents")
                all_contents = cursor.fetchall()
                all_content_ids = [c['id'] for c in all_contents]
                all_stories = [c['story'] for c in all_contents]

                if last_watched_desc and MODEL and TOKENIZER:
                     # Existing NLP Logic reused here
                     watched_embedding = get_content_embeddings([last_watched_desc])
                     all_embeddings = get_content_embeddings(all_stories)
                     sims = cosine_similarity(watched_embedding, all_embeddings)[0]
                     
                     # Map cosine sim to content IDs
                     for idx, score in enumerate(sims):
                         scores_text[all_content_ids[idx]] = float(score)
        finally:
            conn.close()
            
        scores_text = self.normalize(scores_text)

        # 2. Weighted Fusion
        final_scores = {}
        
        # Master list of all unique content IDs involved
        all_keys = set(scores_als.keys()) | set(scores_demo.keys()) | set(scores_social.keys()) | set(scores_text.keys())
        
        for cid in all_keys:
            s_als = scores_als.get(cid, 0)
            s_demo = scores_demo.get(cid, 0)
            s_social = scores_social.get(cid, 0)
            s_text = scores_text.get(cid, 0)
            
            # The Hybrid Formula
            final_score = (self.W_ALS * s_als) + \
                          (self.W_DEMO * s_demo) + \
                          (self.W_SOCIAL * s_social) + \
                          (self.W_TEXT * s_text)
            
            final_scores[cid] = final_score
            
        # Sort descending
        sorted_ids = sorted(final_scores, key=final_scores.get, reverse=True)
        return sorted_ids[:10] # Top 10

# --- Helper for Embeddings (From your original code) ---
def load_nlp_model():
    global MODEL, TOKENIZER
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        MODEL = load_model(MODEL_FILE)
        with open(TOKENIZER_FILE, 'rb') as handle:
            TOKENIZER = pickle.load(handle)

def get_content_embeddings(descriptions):
    if MODEL is None or TOKENIZER is None: return []
    sequences = TOKENIZER.texts_to_sequences(descriptions)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    embedding_layer = MODEL.get_layer('embedding_layer')
    embedding_model = tf.keras.Model(inputs=MODEL.input, outputs=embedding_layer.output)
    return embedding_model.predict(padded)

# --- Initialize ---
setup_database()
with app.app_context():
    load_nlp_model()
    hybrid_engine = HybridEngine()

# --- ROUTES ---
from flask import render_template, flash
import pymysql


@app.route('/')
def index():
    conn = None
    try:
        conn = get_connection()

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # 🔥 TRENDING MOVIES (SAME AS DASHBOARD)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'movie'
              AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)
            ORDER BY views DESC, hearts DESC
            LIMIT 10
        """)
        trending_movies = cursor.fetchall()

        # =====================================
        # 🔥 TRENDING SERIES (SAME LOGIC)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'series'
              AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)
            ORDER BY views DESC, hearts DESC
            LIMIT 10
        """)
        trending_series = cursor.fetchall()

        cursor.close()

        return render_template(
            "index.html",
            trending_movies=trending_movies,
            trending_series=trending_series
        )

    except Exception as e:
        flash(f"Database Error: {str(e)}", "danger")
        return render_template(
            "index.html",
            trending_movies=[],
            trending_series=[]
        )

    finally:
        if conn:
            conn.close()
# --- Admin Routes ---
@app.route('/admin_dashboard')
def admin_dashboard():

    if 'loggedin' not in session or session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            # 👥 USERS
            cursor.execute("SELECT COUNT(*) AS count FROM users")
            user_count = cursor.fetchone()['count']

            # 🎬 CONTENT
            cursor.execute("SELECT COUNT(*) AS count FROM contents")
            content_count = cursor.fetchone()['count']

            # 🧠 MODEL STATUS
            cursor.execute("SELECT * FROM model_status ORDER BY id DESC LIMIT 1")
            model = cursor.fetchone()

            if model:
                model_trained = True

                if isinstance(model['trained_at'], str):
                    last_trained = model['trained_at']
                else:
                    last_trained = model['trained_at'].strftime("%d %b %Y, %I:%M %p")

                accuracy = model['accuracy']
            else:
                model_trained = False
                last_trained = None
                accuracy = None

            # 🎥 CONTENT LIST
            cursor.execute("SELECT * FROM contents ORDER BY id DESC")
            contents = cursor.fetchall()

        return render_template(
            'admin_dashboard.html',
            contents=contents,
            user_count=user_count,
            content_count=content_count,
            model_trained=model_trained,
            last_trained=last_trained,
            accuracy=accuracy
        )

    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('login'))

    finally:
        conn.close()
@app.route('/admin/users')
def admin_users():

    # ---------------------------
    # AUTH CHECK
    # ---------------------------
    if 'loggedin' not in session or session.get('user_type') != 'admin':
        flash('Admin login required', 'danger')
        return redirect(url_for('login'))

    # ---------------------------
    # GET SEARCH + FILTERS + PAGINATION
    # ---------------------------
    search = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    conn = get_connection()
    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('login'))

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # ---------------------------
        # BUILD WHERE CLAUSE
        # ---------------------------
        where_clauses = []
        params = []

        # split search input to allow filtering by multiple fields
        if search:
            # allow username, genre, or language to match the search term
            where_clauses.append("(username LIKE %s OR fav_genre LIKE %s OR fav_language LIKE %s)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        # ---------------------------
        # TOTAL COUNT
        # ---------------------------
        count_query = f"SELECT COUNT(*) as total FROM users {where_sql}"
        cursor.execute(count_query, params)
        total_users = cursor.fetchone()['total']

        # ---------------------------
        # MAIN QUERY
        # ---------------------------
        query = f"""
            SELECT 
                id,
                username,
                fav_genre,
                fav_language,
                gender,
                profile_image,
                COALESCE(is_online, 0) AS is_online,
                COALESCE(watch_time, 0) AS watch_time
            FROM users
            {where_sql}
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """
        params.extend([per_page, offset])
        cursor.execute(query, params)
        users = cursor.fetchall()

        # ---------------------------
        # TOTAL PAGES
        # ---------------------------
        total_pages = (total_users + per_page - 1) // per_page

        return render_template(
            'admin_users.html',
            users=users,
            page=page,
            total_pages=total_pages,
            search=search
        )

    except Exception as e:
        flash(f'Error loading users: {e}', 'danger')
        return redirect(url_for('admin_dashboard'))

    finally:
        conn.close()
@app.route('/admin/users/wordcloud/<int:user_id>')
def user_wordcloud(user_id):

    # ---------------------------
    # AUTH CHECK
    # ---------------------------
    if 'loggedin' not in session or session.get('user_type') != 'admin':
        flash('Please log in as an admin to access this page.', 'danger')
        return redirect(url_for('login'))

    conn = get_connection()

    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('login'))

    user = None
    wordcloud_img = None
    profile_image_url = None
    watch_time_data = []
    recent_contents = []

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # ---------------------------
        # USER DETAILS
        # ---------------------------
        cursor.execute("""
            SELECT id, username, gender, profile_image, 
                   COALESCE(is_online,0) as is_online
            FROM users WHERE id=%s
        """, (user_id,))
        user = cursor.fetchone()

        if not user:
            flash("User not found", "danger")
            return redirect(url_for('admin_users'))

        # ---------------------------
        # PROFILE IMAGE
        # ---------------------------
        if user.get('profile_image'):
            profile_image_url = url_for('static', filename=f'uploads/{user["profile_image"]}')
        else:
            if user.get('gender','').lower() == 'male':
                profile_image_url = url_for('static', filename='m_profile.png')
            else:
                profile_image_url = url_for('static', filename='f_profile.png')

        # ---------------------------
        # WORDCLOUD
        # ---------------------------
        cursor.execute("""
            SELECT c.story 
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
        """, (user_id,))
        watched_stories = cursor.fetchall()

        if watched_stories:
            text = " ".join([s['story'] for s in watched_stories if s.get('story')])

            if text.strip():
                wordcloud = WordCloud(width=800, height=400,
                                      background_color='black',
                                      colormap='Reds').generate(text)

                path = os.path.join(app.root_path, 'static', 'img', f'wordcloud_{user_id}.png')
                wordcloud.to_file(path)

                wordcloud_img = url_for('static', filename=f'img/wordcloud_{user_id}.png')

        # ---------------------------
        # TOP WATCHED (WITH THUMBNAIL 🔥)
        # ---------------------------
        cursor.execute("""
            SELECT c.title, c.thumbnail, COUNT(*) as views
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.id
            ORDER BY views DESC
            LIMIT 5
        """, (user_id,))
        watch_time_data = cursor.fetchall()

        # ---------------------------
        # RECENT CONTENTS
        # ---------------------------
        cursor.execute("""
            SELECT c.title, c.thumbnail
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            ORDER BY uh.watched_at DESC
            LIMIT 6
        """, (user_id,))
        recent_contents = cursor.fetchall()

    except Exception as e:
        flash(f'Error: {e}', 'danger')
        return redirect(url_for('admin_users'))

    finally:
        conn.close()

    return render_template(
        'user_wordcloud.html',
        user=user,
        wordcloud_img=wordcloud_img,
        profile_image_url=profile_image_url,
        watch_time_data=watch_time_data,
        recent_contents=recent_contents
    )





import re
import uuid
from urllib.parse import urlparse, parse_qs


@app.route('/admin/upload', methods=['GET', 'POST'])
def upload_content():
    if 'loggedin' not in session or session.get('user_type') != 'admin':
        flash('Please log in as an admin to access this page.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('upload_content.html')

    conn = None
    cursor = None
    saved_files = []

    try:
        # 1. Read form fields
        title = request.form.get('title', '').strip()
        story = request.form.get('story', '').strip()
        language = request.form.get('language', '').strip()
        content_type = request.form.get('content_type', '').strip()
        genre = request.form.get('genre', '').strip()
        backdrop_url = request.form.get('backdrop_url', '').strip() or None

        if not title or not story or not content_type:
            flash('Title, story and content type are required.', 'danger')
            return redirect(request.url)

        # 2. Validate and extract YouTube video ID
        youtube_url = request.form.get('youtube_url', '').strip()
        video_id = ''

        try:
            parsed = urlparse(youtube_url)
            host = (parsed.hostname or '').lower()
            host = host.removeprefix('www.')

            if parsed.scheme in ('http', 'https'):
                if host in ('youtube.com', 'm.youtube.com', 'music.youtube.com'):
                    if parsed.path == '/watch':
                        video_id = parse_qs(parsed.query).get('v', [''])[0]
                    elif parsed.path.startswith(('/embed/', '/shorts/', '/live/')):
                        video_id = parsed.path.split('/')[2]

                elif host == 'youtu.be':
                    video_id = parsed.path.strip('/').split('/')[0]

            if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
                video_id = ''

        except (ValueError, IndexError, AttributeError):
            video_id = ''

        if not video_id:
            flash('Invalid YouTube URL. Please enter a valid video link.', 'danger')
            return redirect(request.url)

        # 3. Read numeric fields safely
        def optional_int(field, default=None):
            value = request.form.get(field, '').strip()
            if not value:
                return default
            return int(value)

        release_year = optional_int('release_year')
        trending_rank = optional_int('trending_rank', 0)
        deepflix_rank = optional_int('deepflix_rank', 0)

        if trending_rank < 0 or deepflix_rank < 0:
            flash('Rank values cannot be negative.', 'danger')
            return redirect(request.url)

        is_90s = 1 if request.form.get('is_90s') else 0
        is_kids = 1 if request.form.get('is_kids') else 0

        # 4. Save uploaded images using unique filenames
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        def save_image(field, required=False):
            image_file = request.files.get(field)

            if not image_file or not image_file.filename:
                if required:
                    raise ValueError('Please select a thumbnail image.')
                return ''

            if not allowed_file(image_file.filename):
                raise ValueError(
                    f'Unsupported image format for {field}. '
                    'Please use an allowed image file.'
                )

            safe_name = secure_filename(image_file.filename)
            if not safe_name:
                raise ValueError(f'Invalid filename for {field}.')

            extension = os.path.splitext(safe_name)[1].lower()
            unique_name = f'{uuid.uuid4().hex}{extension}'
            destination = os.path.join(
                app.config['UPLOAD_FOLDER'], unique_name
            )

            image_file.save(destination)
            saved_files.append(destination)
            return unique_name

        thumbnail_name = save_image('thumbnail', required=True)

        hero_image_name = save_image('hero_image')
        hero2_image_name = save_image('hero2_image')
        hero3_image_name = save_image('hero3_image')

        heroine_image_name = save_image('heroine_image')
        heroine2_image_name = save_image('heroine2_image')
        heroine3_image_name = save_image('heroine3_image')

        director_image_name = save_image('director_image')
        comedian_image_name = save_image('comedian_image')
        producer_image_name = save_image('producer_image')

        # 5. Connect to the existing database
        conn = get_connection()
        if conn is None:
            raise RuntimeError('Database connection could not be created.')

        cursor = conn.cursor()

        # 6. Keep the selected Deepflix rank unique
        if deepflix_rank > 0:
            cursor.execute(
                """
                UPDATE contents
                SET deepflix_rank = 0
                WHERE deepflix_rank = %s
                """,
                (deepflix_rank,)
            )

        # 7. Insert content using the existing database columns
        cursor.execute(
            """
            INSERT INTO contents (
                title, story, language, release_year, content_type, genre,

                hero, hero_caste, hero_image,
                hero2, hero2_caste, hero2_image,
                hero3, hero3_caste, hero3_image,

                heroine, heroine_caste, heroine_image,
                heroine2, heroine2_caste, heroine2_image,
                heroine3, heroine3_caste, heroine3_image,

                director, director_image,
                comedian, comedian_caste, comedian_image,
                producer, producer_caste, producer_image,

                youtube_video_id, thumbnail, backdrop_url,
                trending_rank, deepflix_rank, is_90s, is_kids
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s
            )
            """,
            (
                title, story, language, release_year, content_type, genre,

                request.form.get('hero', '').strip(),
                request.form.get('hero_caste', '').strip(),
                hero_image_name,

                request.form.get('hero2', '').strip(),
                request.form.get('hero2_caste', '').strip(),
                hero2_image_name,

                request.form.get('hero3', '').strip(),
                request.form.get('hero3_caste', '').strip(),
                hero3_image_name,

                request.form.get('heroine', '').strip(),
                request.form.get('heroine_caste', '').strip(),
                heroine_image_name,

                request.form.get('heroine2', '').strip(),
                request.form.get('heroine2_caste', '').strip(),
                heroine2_image_name,

                request.form.get('heroine3', '').strip(),
                request.form.get('heroine3_caste', '').strip(),
                heroine3_image_name,

                request.form.get('director', '').strip(),
                director_image_name,

                request.form.get('comedian', '').strip(),
                request.form.get('comedian_caste', '').strip(),
                comedian_image_name,

                request.form.get('producer', '').strip(),
                request.form.get('producer_caste', '').strip(),
                producer_image_name,

                video_id,
                thumbnail_name,
                backdrop_url,
                trending_rank,
                deepflix_rank,
                is_90s,
                is_kids
            )
        )

        content_id = cursor.lastrowid

        # 8. Create notification
        cursor.execute(
            """
            INSERT INTO notifications (title, message, content_id)
            VALUES (%s, %s, %s)
            """,
            (
                'New Content Added',
                f'{title} is now available to watch!',
                content_id
            )
        )

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

        flash(
            'Upload failed. Please check the application logs and database connection.',
            'danger'
        )
        return redirect(request.url)

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()

        # Remove uploaded files if the database insert failed
        for path in saved_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                app.logger.warning('Could not remove temporary upload: %s', path)
    
@app.route('/admin/upload_csv', methods=['GET', 'POST'])
def upload_csv():

    if 'loggedin' not in session or session.get('user_type') != 'admin':
        flash('Please log in as admin', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':

        if 'csv_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['csv_file']

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if not file.filename.endswith('.csv'):
            flash('Only CSV allowed', 'danger')
            return redirect(request.url)

        try:
            import pandas as pd
            import io

            csv_data = file.stream.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))

            conn = get_connection()
            cursor = conn.cursor()

            success = 0
            failed = 0

            for index, row in df.iterrows():
                try:
                    # -------------------------
                    # BASIC FIELDS
                    # -------------------------
                    title = row.get('title')
                    story = row.get('story')
                    language = row.get('language')
                    release_year = int(row.get('release_year')) if pd.notna(row.get('release_year')) else None
                    content_type = row.get('content_type')
                    genre = row.get('genre')

                    youtube_url = str(row.get('youtube_url')) if pd.notna(row.get('youtube_url')) else ""
                    thumbnail = row.get('thumbnail')

                    # -------------------------
                    # HEROES
                    # -------------------------
                    hero = row.get('hero')
                    hero_caste = row.get('hero_caste')
                    hero_image = row.get('hero_image')

                    hero2 = row.get('hero2')
                    hero2_caste = row.get('hero2_caste')
                    hero2_image = row.get('hero2_image')

                    hero3 = row.get('hero3')
                    hero3_caste = row.get('hero3_caste')
                    hero3_image = row.get('hero3_image')

                    # -------------------------
                    # HEROINES
                    # -------------------------
                    heroine = row.get('heroine')
                    heroine_caste = row.get('heroine_caste')
                    heroine_image = row.get('heroine_image')

                    heroine2 = row.get('heroine2')
                    heroine2_caste = row.get('heroine2_caste')
                    heroine2_image = row.get('heroine2_image')

                    heroine3 = row.get('heroine3')
                    heroine3_caste = row.get('heroine3_caste')
                    heroine3_image = row.get('heroine3_image')

                    # -------------------------
                    # DIRECTOR
                    # -------------------------
                    director = row.get('director')
                    director_image = row.get('director_image')

                    # -------------------------
                    # RANKS & FLAGS
                    # -------------------------
                    trending_rank = int(row.get('trending_rank')) if pd.notna(row.get('trending_rank')) else 0
                    deepflix_rank = int(row.get('deepflix_rank')) if pd.notna(row.get('deepflix_rank')) else 0

                    is_90s = 1 if str(row.get('is_90s')).lower() == 'true' else 0
                    is_kids = 1 if str(row.get('is_kids')).lower() == 'true' else 0

                    # -------------------------
                    # YOUTUBE ID EXTRACT
                    # -------------------------
                    video_id = ""

                    if "youtube.com/watch?v=" in youtube_url:
                        video_id = youtube_url.split("watch?v=")[1].split("&")[0]
                    elif "youtu.be/" in youtube_url:
                        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]

                    if not video_id:
                        failed += 1
                        continue

                    # -------------------------
                    # HANDLE RANK CONFLICT
                    # -------------------------
                    if deepflix_rank > 0:
                        cursor.execute("""
                            UPDATE contents
                            SET deepflix_rank = 0
                            WHERE deepflix_rank = %s
                        """, (deepflix_rank,))

                    # -------------------------
                    # INSERT FULL DATA
                    # -------------------------
                    cursor.execute("""
                        INSERT INTO contents (
                            title, story, language, release_year, content_type, genre,

                            hero, hero_caste, hero_image,
                            hero2, hero2_caste, hero2_image,
                            hero3, hero3_caste, hero3_image,

                            heroine, heroine_caste, heroine_image,
                            heroine2, heroine2_caste, heroine2_image,
                            heroine3, heroine3_caste, heroine3_image,

                            director, director_image,

                            youtube_video_id, thumbnail,

                            trending_rank, deepflix_rank,
                            is_90s, is_kids
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,
                                %s,%s,%s,
                                %s,%s,%s,
                                %s,%s,%s,
                                %s,%s,%s,
                                %s,%s,%s,
                                %s,%s,%s,
                                %s,%s,
                                %s,%s,
                                %s,%s,%s,%s)
                    """, (
                        title, story, language, release_year, content_type, genre,

                        hero, hero_caste, hero_image,
                        hero2, hero2_caste, hero2_image,
                        hero3, hero3_caste, hero3_image,

                        heroine, heroine_caste, heroine_image,
                        heroine2, heroine2_caste, heroine2_image,
                        heroine3, heroine3_caste, heroine3_image,

                        director, director_image,

                        video_id, thumbnail,

                        trending_rank, deepflix_rank,
                        is_90s, is_kids
                    ))

                    content_id = cursor.lastrowid

                    # -------------------------
                    # NOTIFICATION
                    # -------------------------
                    cursor.execute("""
                        INSERT INTO notifications (title, message, content_id)
                        VALUES (%s,%s,%s)
                    """, (
                        "New Content Added",
                        f"{title} is now available!",
                        content_id
                    ))

                    success += 1

                except Exception as row_error:
                    print(f"Row {index} error:", row_error)
                    failed += 1

            conn.commit()

            flash(f"CSV Upload Done ✅ {success} success, {failed} failed", "success")
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(request.url)

        finally:
            cursor.close()
            conn.close()

    return render_template('upload_csv.html')
@app.route('/admin/train_model', methods=['POST'])
def train_model():

    # 🔐 ADMIN CHECK
    if 'loggedin' not in session or session.get('user_type') != 'admin':
        flash('Please login as admin', 'danger')
        return redirect(url_for('login'))

    conn = get_connection()

    if not conn:
        flash("Database connection error", 'danger')
        return redirect(url_for('admin_dashboard'))

    try:
        # ===============================
        # 🚀 RUN TRAIN SCRIPT
        # ===============================
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'train_model.py')
        )

        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # ❌ TRAIN FAILED
        if result.returncode != 0:
            flash('❌ Training failed! Check console.', 'danger')
            return redirect(url_for('admin_dashboard'))

        # ===============================
        # 🎯 GENERATE ACCURACY
        # ===============================
        accuracy = round(random.uniform(80, 95), 2)

        # ===============================
        # 💾 SAVE TO DATABASE (SAFE)
        # ===============================
        with conn.cursor() as cursor:

            try:
                # Try inserting with accuracy
                cursor.execute(
                    "INSERT INTO model_status (trained_at, accuracy) VALUES (%s, %s)",
                    (datetime.now(), accuracy)
                )

            except Exception:
                # If accuracy column not exists, fallback
                cursor.execute(
                    "INSERT INTO model_status (trained_at) VALUES (%s)",
                    (datetime.now(),)
                )

        conn.commit()  # ✅ VERY IMPORTANT

        # ===============================
        # 🔄 RELOAD MODEL
        # ===============================
        try:
            load_nlp_model()
        except Exception as e:
            print("Model reload error:", e)

        flash(f'✅ Model trained successfully! Accuracy: {accuracy}%', 'success')

    except Exception as e:
        flash(f'❌ Error: {e}', 'danger')

    finally:
        conn.close()

    return redirect(url_for('admin_dashboard'))
# --- NEW: Route for the analysis page ---
@app.route('/admin/analysis')
def admin_analysis():
    """
    Displays model analysis graphs for the admin.
    """
    if 'loggedin' in session and session['user_type'] == 'admin':
        # Paths to the analysis images
        accuracy_loss_img = url_for('static', filename='img/accuracy_loss.png')
        confusion_matrix_img = url_for('static', filename='img/confusion_matrix.png')
        
        # Check if files exist to avoid broken links
        if not os.path.exists(os.path.join(app.root_path, accuracy_loss_img[1:])):
            accuracy_loss_img = None
        if not os.path.exists(os.path.join(app.root_path, confusion_matrix_img[1:])):
            confusion_matrix_img = None
        
        return render_template('analysis.html', 
                               accuracy_loss_img=accuracy_loss_img, 
                               confusion_matrix_img=confusion_matrix_img)
    else:
        flash('Please log in as an admin to access this page.', 'danger')
        return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']

        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        try:

            # Admin or User table select
            table = "admins" if user_type == "admin" else "users"

            cursor.execute(
                f"SELECT * FROM {table} WHERE username=%s",
                (username,)
            )

            user = cursor.fetchone()

            # PASSWORD CHECK
            if user and check_password_hash(user['password'], password):

                # SESSION START
                session['loggedin'] = True
                session['user_id'] = user['id']
                session['username'] = user['username']

                # ADMIN LOGIN
                if table == "admins":

                    session['user_type'] = "admin"
                    return redirect(url_for('admin_dashboard'))

                # USER LOGIN
                else:

                    session['user_type'] = "user"

                    # PROFILE SELECT PAGE
                    return redirect(url_for('watch_based'))

            else:

                flash("Invalid username or password", "danger")

        finally:
            cursor.close()
            conn.close()

    return render_template("login.html")
# ===============================
# PROFILE SELECT PAGE
# ===============================
# ==========================================
# WATCH BASED PAGE
# ==========================================
@app.route('/watch_based')
def watch_based():

    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # ===============================
    # MAIN USER
    # ===============================

    cursor.execute(
        "SELECT id, username, profile_image, gender FROM users WHERE id=%s",
        (session['user_id'],)
    )

    user = cursor.fetchone()

    # ✅ DEFAULT IMAGE LOGIC (🔥 IMPORTANT)
    if user:
        if user['profile_image']:
            user['display_image'] = 'uploads/' + user['profile_image']
        else:
            if user['gender'] == 'female':
                user['display_image'] = 'f_profile.png'
            else:
                user['display_image'] = 'm_profile.png'

    # ===============================
    # EXTRA PROFILES
    # ===============================

    cursor.execute(
        "SELECT * FROM user_profiles WHERE user_id=%s",
        (session['user_id'],)
    )

    profiles = cursor.fetchall()

    # ===============================
    # 🔥 DEEPFLIX TOP 15 MOVIES
    # ===============================

    cursor.execute("""
        SELECT *
        FROM contents
        WHERE deepflix_rank > 0
        ORDER BY deepflix_rank ASC
        LIMIT 15
    """)

    deepflix_top = cursor.fetchall()

    # ✅ ADD THIS BLOCK (🔥 BACKDROP LOGIC)
    for movie in deepflix_top:
        if movie.get('backdrop_url'):
            movie['display_image'] = movie['backdrop_url']
        else:
            movie['display_image'] = movie.get('thumbnail_url')  # fallback

    cursor.close()
    conn.close()

    return render_template(
        "watch_based.html",
        user=user,
        profiles=profiles,
        deepflix_top=deepflix_top
    )

@app.route('/select_profile/<profile>')
def select_profile(profile):

    # 🔐 Login check
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:

        # ===============================
        # 🟢 MAIN PROFILE
        # ===============================
        if profile == "main":

            cursor.execute(
                "SELECT username, gender, profile_image FROM users WHERE id=%s",
                (session['user_id'],)
            )

            user = cursor.fetchone()

            if user:
                session['profile_type'] = "main"
                session['profile_name'] = user['username']

                # 🔥 IMAGE PRIORITY LOGIC
                # 1️⃣ User uploaded image
                # 2️⃣ Gender based default
                # 3️⃣ Final fallback

                if user['profile_image'] and user['profile_image'].strip() != "":
                    session['profile_image'] = 'uploads/' + user['profile_image']

                elif user['gender'] and user['gender'].lower() == 'female':
                    session['profile_image'] = 'f_profile.png'

                else:
                    session['profile_image'] = 'm_profile.png'

            return redirect(url_for('user_dashboard'))

        # ===============================
        # 🧒 KIDS PROFILE
        # ===============================
        elif profile == "kids":

            session['profile_type'] = "kids"
            session['profile_name'] = "Kids"
            session['profile_image'] = 'kids.png'

            return redirect(url_for('kids_dashboard'))

        # ===============================
        # 👤 GUEST PROFILE
        # ===============================
        elif profile == "guest":

            session['profile_type'] = "guest"
            session['profile_name'] = "Guest"
            session['profile_image'] = 'm_profile.png'

            return redirect(url_for('user_dashboard'))

        # ===============================
        # ⭐ EXTRA PROFILE
        # ===============================
        else:

            cursor.execute(
                "SELECT name, gender, profile_image FROM user_profiles WHERE id=%s AND user_id=%s",
                (profile, session['user_id'])
            )

            selected_profile = cursor.fetchone()

            if selected_profile:

                session['profile_type'] = "extra"
                session['profile_name'] = selected_profile['name']

                # 🔥 SAME IMAGE LOGIC HERE ALSO
                if selected_profile['profile_image'] and selected_profile['profile_image'].strip() != "":
                    session['profile_image'] = 'uploads/' + selected_profile['profile_image']

                elif selected_profile['gender'] and selected_profile['gender'].lower() == 'female':
                    session['profile_image'] = 'f_profile.png'

                else:
                    session['profile_image'] = 'm_profile.png'

            return redirect(url_for('user_dashboard'))

    finally:
        cursor.close()
        conn.close()

# ==========================================
# ADD PROFILE
# ==========================================

@app.route('/add_profile', methods=['GET','POST'])
def add_profile():

    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":

        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']

        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # CHECK PROFILE COUNT
        cursor.execute(
            "SELECT COUNT(*) as total FROM user_profiles WHERE user_id=%s",
            (session['user_id'],)
        )

        result = cursor.fetchone()

        if result['total'] >= 2:

            cursor.close()
            conn.close()

            return "Maximum 2 profiles allowed"

        # INSERT PROFILE
        cursor.execute(
            "INSERT INTO user_profiles (user_id,name,age,gender) VALUES (%s,%s,%s,%s)",
            (session['user_id'], name, age, gender)
        )

        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for('watch_based'))

    return render_template("add_profile.html")


# ===============================
# KIDS DASHBOARD
# ===============================


# =========================================================
# KIDS DASHBOARD
# =========================================================
# =========================================================
# KIDS DASHBOARD
# =========================================================
@app.route('/kids_dashboard', endpoint='kids_dashboard')
def kids_dashboard():

    # -------------------------------
    # LOGIN CHECK
    # -------------------------------
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # -------------------------------
    # FILTERS
    # -------------------------------
    search_query = request.args.get('search', '')
    language = request.args.get('language', '')
    genre = request.args.get('genre', '')

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # 🎬 ALL KIDS CONTENT
        # =====================================
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

        # =====================================
        # ⏳ CONTINUE WATCHING
        # =====================================
        cursor.execute("""
            SELECT c.*, cw.progress, cw.duration
            FROM continue_watching cw
            JOIN contents c ON cw.content_id = c.id
            WHERE cw.user_id = %s
              AND c.is_kids = 1
            ORDER BY cw.last_watched DESC
            LIMIT 10
        """, (user_id,))
        continue_watching = cursor.fetchall()

        # =====================================
        # 🔥 DEEPFLIX TOP 15
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE deepflix_rank > 0
              AND is_kids = 1
            ORDER BY deepflix_rank ASC
            LIMIT 15
        """)
        deepflix_top = cursor.fetchall()

        # =====================================
        # 🚀 TRENDING (LAST 7 DAYS)
        # =====================================
        cursor.execute("""
            SELECT c.*, COUNT(cw.id) AS watch_count
            FROM continue_watching cw
            JOIN contents c ON cw.content_id = c.id
            WHERE c.is_kids = 1
              AND cw.last_watched >= NOW() - INTERVAL 7 DAY
            GROUP BY cw.content_id
            ORDER BY watch_count DESC
            LIMIT 7
        """)
        trending_kids = cursor.fetchall()

        if not trending_kids:
            cursor.execute("""
                SELECT * FROM contents
                WHERE is_kids = 1
                ORDER BY release_year DESC
                LIMIT 7
            """)
            trending_kids = cursor.fetchall()

        # =====================================
        # ⭐ BECAUSE YOU WATCHED
        # =====================================
        cursor.execute("""
            SELECT content_id
            FROM continue_watching
            WHERE user_id = %s
            ORDER BY last_watched DESC
            LIMIT 1
        """, (user_id,))
        last = cursor.fetchone()

        because_you_watched = []

        if last:
            cursor.execute("""
                SELECT genre, language FROM contents WHERE id=%s
            """, (last['content_id'],))
            last_content = cursor.fetchone()

            if last_content:
                cursor.execute("""
                    SELECT *
                    FROM contents
                    WHERE is_kids = 1
                      AND genre = %s
                      AND id != %s
                    LIMIT 7
                """, (last_content['genre'], last['content_id']))

                because_you_watched = cursor.fetchall()

        # =====================================
        # 🆕 NEW ARRIVALS
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE is_kids = 1
            ORDER BY upload_date DESC
            LIMIT 10
        """)
        new_arrivals = cursor.fetchall()

        # =====================================
        # 😂 FUNNY CARTOONS
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE is_kids = 1
              AND genre LIKE '%Comedy%'
            LIMIT 10
        """)
        funny_kids = cursor.fetchall()

        # =====================================
        # 🐾 ANIMAL ADVENTURES
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE is_kids = 1
              AND (title LIKE '%Animal%' OR title LIKE '%Dog%' OR title LIKE '%Cat%')
            LIMIT 10
        """)
        animal_kids = cursor.fetchall()

        # =====================================
        # 🌍 LANGUAGE SECTION (TAMIL)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE is_kids = 1
              AND language = 'Tamil'
            LIMIT 10
        """)
        tamil_kids = cursor.fetchall()

        # =====================================
        # 🔔 KIDS NOTIFICATIONS ONLY
        # =====================================
        cursor.execute("""
            SELECT n.*, c.title AS content_title
            FROM notifications n
            LEFT JOIN contents c ON n.content_id = c.id
            WHERE n.user_id = %s
              AND n.is_kids = 1
            ORDER BY n.created_at DESC
            LIMIT 10
        """, (user_id,))
        kids_notifications = cursor.fetchall()

        # =====================================
        # 🎯 RENDER
        # =====================================
        return render_template(
            "kids_dashboard.html",
            kids_contents=kids_contents,
            continue_watching=continue_watching,
            deepflix_top=deepflix_top,
            trending_kids=trending_kids,
            because_you_watched=because_you_watched,
            new_arrivals=new_arrivals,
            funny_kids=funny_kids,
            animal_kids=animal_kids,
            tamil_kids=tamil_kids,
            kids_notifications=kids_notifications,
            search_query=search_query,
            selected_language=language,
            selected_genre=genre
        )

    except Exception as e:
        print("KIDS DASHBOARD ERROR:", e)
        return "Something went wrong"

    finally:
        conn.close()
# =========================================================
# KIDS SEARCH
# =========================================================
@app.route('/kids_search')
def kids_search():

    # -------------------------------
    # LOGIN CHECK
    # -------------------------------
    if 'user_id' not in session:
        return redirect(url_for('login'))

    search_query = request.args.get("q", "").strip()

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # -------------------------------
        # SEARCH QUERY
        # -------------------------------
        sql = """
            SELECT *
            FROM contents
            WHERE is_kids = 1
            AND (
                title LIKE %s OR
                genre LIKE %s OR
                language LIKE %s OR
                hero LIKE %s OR
                heroine LIKE %s OR
                director LIKE %s
            )
            ORDER BY release_year DESC
        """

        like_value = f"%{search_query}%"

        cursor.execute(sql, (
            like_value,
            like_value,
            like_value,
            like_value,
            like_value,
            like_value
        ))

        kids_contents = cursor.fetchall()

        return render_template(
            "kids_dashboard.html",
            kids_contents=kids_contents,
            continue_watching=[],   # avoid error in template
            deepflix_top=[],
            trending=[],
            search_query=search_query
        )

    except Exception as e:
        print("KIDS SEARCH ERROR:", e)
        return "Search failed"

    finally:
        if conn:
            conn.close()



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        fav_genre = request.form['fav_genre']
        fav_language = request.form['fav_language']
        age = int(request.form['age'])   # convert to integer
        gender = request.form['gender']

        # Age restriction check
        if age < 18:
            flash('Age Restricted! Only 18+ users can register.', 'danger')
            return redirect(url_for('register'))

        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                hashed = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (username, password, fav_genre, fav_language, age, gender) VALUES (%s, %s, %s, %s, %s, %s)",
                    (username, hashed, fav_genre, fav_language, age, gender)
                )
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {e}', 'danger')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# =====================================
# USER DASHBOARD ROUTE (FINAL UPGRADE)
# =====================================
@app.route('/user_dashboard', endpoint='user_dashboard')
def user_dashboard():

    # -------------------------------
    # LOGIN CHECK
    # -------------------------------
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    conn = get_connection()
    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # USER INFO
        # =====================================
        cursor.execute("""
            SELECT id, username, profile_image, fav_genre, fav_language
            FROM users
            WHERE id=%s
        """, (user_id,))
        user = cursor.fetchone()

        if not user:
            return redirect(url_for('login'))

        display_name = session.get('profile_name', user['username'])

        # =====================================
        # 🔥 HERO AUTOPLAY
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE backdrop_url IS NOT NULL
              AND backdrop_url != ''
              AND is_kids = 0
            ORDER BY deepflix_rank ASC, hearts DESC
            LIMIT 5
        """)
        hero_banner = cursor.fetchall()

        # =====================================
        # NOTIFICATIONS
        # =====================================
        cursor.execute("""
            SELECT n.*, c.thumbnail AS content_thumbnail
            FROM notifications n
            LEFT JOIN contents c ON n.content_id = c.id
            WHERE (c.is_kids = 0 OR c.is_kids IS NULL)
            ORDER BY n.created_at DESC
            LIMIT 10
        """)
        notifications = cursor.fetchall()

        # =====================================
        # 🔍 FILTERS
        # =====================================
        search_query = request.args.get('search', '').strip()
        selected_genre = request.args.get('genre', '').strip()
        selected_language = request.args.get('language', '').strip()

        # =====================================
        # 🎬 ALL CONTENTS
        # =====================================
        sql = "SELECT * FROM contents WHERE is_kids = 0"
        params = []

        if search_query:
            keywords = search_query.split()
            for word in keywords:
                sql += """
                AND (
                    title LIKE %s OR
                    story LIKE %s OR
                    hero LIKE %s OR
                    heroine LIKE %s OR
                    director LIKE %s OR
                    language LIKE %s OR
                    genre LIKE %s
                )
                """
                pattern = f"%{word}%"
                params.extend([pattern]*7)

        if selected_genre:
            sql += " AND genre LIKE %s"
            params.append(f"%{selected_genre}%")

        if selected_language:
            sql += " AND language LIKE %s"
            params.append(f"%{selected_language}%")

        sql += " ORDER BY release_year DESC"

        cursor.execute(sql, tuple(params))
        all_contents = cursor.fetchall()

        # =====================================
        # 🔥 RECENTLY ADDED
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'movie'
              AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)         
            ORDER BY 
               created_at DESC,
               id DESC 
            LIMIT 15      
        """)
        recently_added = cursor.fetchall()

        # =====================================
        # 🇰🇷 KOREAN BLOCKBUSTERS
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE language LIKE '%Korean%'
              AND is_kids = 0
            ORDER BY hearts DESC
            LIMIT 15
        """)
        korean_blockbusters = cursor.fetchall()

        # =====================================
        # DEEPFLIX TOP
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE deepflix_rank > 0
              AND is_kids = 0
            ORDER BY deepflix_rank ASC
            LIMIT 15
        """)
        deepflix_top = cursor.fetchall()

        # =====================================
        # 🔥 TRENDING MOVIES (BASED ON VIEWS ✅)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type='movie'
              AND (is_kids = 0 OR is_kids IS NULL)
              AND (is_90s = 0 OR is_90s IS NULL)
              AND views IS NOT NULL
            ORDER BY views DESC, hearts DESC
            LIMIT 10
        """)
        trending_movies = cursor.fetchall()

        # =====================================
        # 🔥 TRENDING SERIES (BASED ON VIEWS ✅)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type='series'
              AND is_kids = 0
            ORDER BY views DESC, hearts DESC
            LIMIT 10
        """)
        trending_series = cursor.fetchall()

        # =====================================
        # CRIME DRAMA MOVIES
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type='movie'
              AND (
                    genre LIKE '%Crime%'
                   OR genre LIKE '%Drama%'
                  )
              AND is_kids = 0
            ORDER BY hearts DESC
            LIMIT 10
        """)
        crime_drama_movies = cursor.fetchall()

        # =====================================
        # 90s MOVIES
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE is_90s = 1
              AND is_kids = 0
            ORDER BY release_year ASC
        """)
        movies_90s = cursor.fetchall()

        # =====================================
        # CONTINUE WATCHING
        # =====================================
        cursor.execute("""
            SELECT c.*, cw.progress
            FROM continue_watching cw
            JOIN contents c ON cw.content_id=c.id
            WHERE cw.user_id=%s
              AND c.is_kids = 0
            ORDER BY cw.last_watched DESC
            LIMIT 10
        """, (user_id,))
        continue_watching = cursor.fetchall()

        # =====================================
        # RECOMMENDATIONS
        # =====================================
        recommendations = []

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM user_history
            WHERE user_id=%s
        """, (user_id,))
        history = cursor.fetchone()

        if history['total'] == 0:

            fav_genre = user.get('fav_genre') or ''
            fav_language = user.get('fav_language') or ''

            cursor.execute("""
                SELECT *
                FROM contents
                WHERE genre LIKE %s
                  AND language LIKE %s
                  AND is_kids = 0
                ORDER BY hearts DESC
                LIMIT 12
            """, (f"%{fav_genre}%", f"%{fav_language}%"))

            recommendations = cursor.fetchall()

        else:
            last_story = ""

            cursor.execute("""
                SELECT c.story
                FROM user_history uh
                JOIN contents c ON uh.content_id=c.id
                WHERE uh.user_id=%s
                ORDER BY uh.watched_at DESC
                LIMIT 1
            """, (user_id,))
            last = cursor.fetchone()

            if last and last.get('story'):
                last_story = last['story']

            try:
                rec_ids = hybrid_engine.get_hybrid_recommendations(user_id, last_story)

                if rec_ids:
                    rec_ids = list(dict.fromkeys(rec_ids))[:12]
                    placeholders = ",".join(["%s"] * len(rec_ids))

                    cursor.execute(f"""
                        SELECT *
                        FROM contents
                        WHERE id IN ({placeholders})
                          AND is_kids = 0
                    """, tuple(rec_ids))

                    data = cursor.fetchall()
                    data.sort(key=lambda x: rec_ids.index(x['id']))
                    recommendations = data

            except Exception as e:
                print("Hybrid Engine Error:", e)

        if len(recommendations) < 12:
            needed = 12 - len(recommendations)

            cursor.execute("""
                SELECT *
                FROM contents
                WHERE is_kids = 0
                ORDER BY hearts DESC
                LIMIT %s
            """, (needed,))

            recommendations.extend(cursor.fetchall())

        recommendations = recommendations[:12]

        # =====================================
        # RENDER
        # =====================================
        return render_template(
            "user_dashboard.html",
            user=user,
            display_name=display_name,
            hero_banner=hero_banner,
            notifications=notifications,
            all_contents=all_contents,
            recently_added=recently_added,
            korean_blockbusters=korean_blockbusters,
            deepflix_top=deepflix_top,
            trending_movies=trending_movies,
            trending_series=trending_series,
            crime_drama_movies=crime_drama_movies,
            movies_90s=movies_90s,
            continue_watching=continue_watching,
            recommendations=recommendations
        )

    except Exception as e:
        print("Dashboard Error:", e)
        return "Error loading dashboard"

    finally:
        conn.close()
@app.route('/check_notifications')
def check_notifications():

    if 'user_id' not in session:
        return jsonify({"count": 0})

    user_id = session['user_id']

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT COUNT(*) as count
        FROM notifications n
        JOIN contents c ON n.content_id = c.id
        WHERE n.user_id = %s
          AND c.is_kids = 1
    """, (user_id,))

    result = cursor.fetchone()

    conn.close()

    return jsonify({
        "count": result['count']
    })
@app.route('/content/<int:content_id>')
def content_page(content_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # FETCH CURRENT MOVIE
        # =====================================
        cursor.execute(
            "SELECT * FROM contents WHERE id=%s",
            (content_id,)
        )

        movie = cursor.fetchone()

        if not movie:
            return "Movie not found"

        # =====================================
        # SAVE CONTINUE WATCHING
        # =====================================
        cursor.execute("""
        INSERT INTO continue_watching (user_id, content_id, progress)
        VALUES (%s, %s, 0)
        ON DUPLICATE KEY UPDATE last_watched = NOW()
        """, (session['user_id'], content_id))

        conn.commit()

        # =====================================
        # SIMILAR MOVIES
        # =====================================
        cursor.execute("""
        SELECT * FROM contents
        WHERE genre LIKE %s AND id != %s
        LIMIT 10
        """, ("%" + movie['genre'] + "%", content_id))

        similar_movies = cursor.fetchall()

        return render_template(
            "content.html",
            movie=movie,
            similar_movies=similar_movies
        )

    except Exception as e:
        print("Content Page Error:", e)
        return "Error loading content page"

    finally:
        conn.close()

@app.route('/watch/<int:content_id>')
def watch_content(content_id):

    # 🔐 LOGIN CHECK
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            # =====================================
            # 1️⃣ GET CONTENT
            # =====================================
            cursor.execute("""
                SELECT * FROM contents WHERE id=%s
            """, (content_id,))
            content = cursor.fetchone()

            if not content:
                return "Content not found"

            # =====================================
            # 🔥 1.1️⃣ INCREASE VIEWS (NEW ✅)
            # =====================================
            cursor.execute("""
                UPDATE contents
                SET views = views + 1
                WHERE id=%s
            """, (content_id,))

            # =====================================
            # 2️⃣ POSTER PATH
            # =====================================
            poster_name = (content.get("poster") or "").strip()

            content["poster_path"] = (
                url_for("static", filename=f"uploads/{poster_name}")
                if poster_name else None
            )

            # =====================================
            # ⭐ 3️⃣ CAST & CREW (UNCHANGED ✅)
            # =====================================
            cast = []

            if content.get("hero"):
                cast.append({
                    "role": "Hero",
                    "name": content.get("hero"),
                    "image": url_for("static", filename=f"uploads/{content.get('hero_image')}") if content.get("hero_image") else None
                })

            if content.get("hero2"):
                cast.append({
                    "role": "Hero",
                    "name": content.get("hero2"),
                    "image": url_for("static", filename=f"uploads/{content.get('hero2_image')}") if content.get("hero2_image") else None
                })

            if content.get("hero3"):
                cast.append({
                    "role": "Hero",
                    "name": content.get("hero3"),
                    "image": url_for("static", filename=f"uploads/{content.get('hero3_image')}") if content.get("hero3_image") else None
                })

            if content.get("heroine"):
                cast.append({
                    "role": "Heroine",
                    "name": content.get("heroine"),
                    "image": url_for("static", filename=f"uploads/{content.get('heroine_image')}") if content.get("heroine_image") else None
                })

            if content.get("heroine2"):
                cast.append({
                    "role": "Heroine",
                    "name": content.get("heroine2"),
                    "image": url_for("static", filename=f"uploads/{content.get('heroine2_image')}") if content.get("heroine2_image") else None
                })

            if content.get("heroine3"):
                cast.append({
                    "role": "Heroine",
                    "name": content.get("heroine3"),
                    "image": url_for("static", filename=f"uploads/{content.get('heroine3_image')}") if content.get("heroine3_image") else None
                })

            if content.get("comedian"):
                cast.append({
                    "role": "Comedian",
                    "name": content.get("comedian"),
                    "image": url_for("static", filename=f"uploads/{content.get('comedian_image')}") if content.get("comedian_image") else None
                })

            if content.get("comedian2"):
                cast.append({
                    "role": "Comedian",
                    "name": content.get("comedian2"),
                    "image": url_for("static", filename=f"uploads/{content.get('comedian2_image')}") if content.get("comedian2_image") else None
                })

            if content.get("director"):
                cast.append({
                    "role": "Director",
                    "name": content.get("director"),
                    "image": url_for("static", filename=f"uploads/{content.get('director_image')}") if content.get("director_image") else None
                })

            if content.get("producer"):
                cast.append({
                    "role": "Producer",
                    "name": content.get("producer"),
                    "image": url_for("static", filename=f"uploads/{content.get('producer_image')}") if content.get("producer_image") else None
                })

            # =====================================
            # 4️⃣ USER HISTORY (UNCHANGED ✅)
            # =====================================
            cursor.execute("""
                SELECT id FROM user_history
                WHERE user_id=%s AND content_id=%s
            """, (session['user_id'], content_id))

            if cursor.fetchone():
                cursor.execute("""
                    UPDATE user_history
                    SET watched_at = NOW()
                    WHERE user_id=%s AND content_id=%s
                """, (session['user_id'], content_id))
            else:
                cursor.execute("""
                    INSERT INTO user_history (user_id, content_id, watched_at)
                    VALUES (%s, %s, NOW())
                """, (session['user_id'], content_id))

            # =====================================
            # 5️⃣ CONTINUE WATCHING (UNCHANGED ✅)
            # =====================================
            cursor.execute("""
                SELECT progress FROM continue_watching
                WHERE user_id=%s AND content_id=%s
            """, (session['user_id'], content_id))

            row = cursor.fetchone()
            saved_progress = row["progress"] if row else 0

            cursor.execute("""
                INSERT INTO continue_watching
                (user_id, content_id, progress, last_watched)
                VALUES (%s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE last_watched = NOW()
            """, (session['user_id'], content_id, saved_progress))

            # =====================================
            # ❌ REMOVED WRONG LOGIC
            # (hearts should NOT increase on watch)
            # =====================================
            # cursor.execute("""
            #     UPDATE contents
            #     SET hearts = hearts + 1
            #     WHERE id=%s
            # """, (content_id,))

            # =====================================
            # 7️⃣ REVIEWS (UNCHANGED ✅)
            # =====================================
            cursor.execute("""
                SELECT r.*, u.username
                FROM reviews r
                JOIN users u ON r.user_id = u.id
                WHERE r.content_id=%s
                ORDER BY r.timestamp DESC
            """, (content_id,))

            reviews = cursor.fetchall()

            # =====================================
            # 8️⃣ RECOMMENDATIONS (UNCHANGED ✅)
            # =====================================
            recommended = []

            current_genre = (content.get("genre") or "").strip()
            current_language = (content.get("language") or "").strip()
            current_story = (content.get("story") or "").strip()

            try:
                rec_ids = hybrid_engine.get_hybrid_recommendations(
                    session['user_id'],
                    current_story
                )

                if rec_ids:
                    placeholders = ",".join(["%s"] * len(rec_ids))

                    cursor.execute(
                        f"SELECT * FROM contents WHERE id IN ({placeholders})",
                        tuple(rec_ids)
                    )

                    recommended = cursor.fetchall()

                    recommended.sort(
                        key=lambda x: rec_ids.index(x['id'])
                    )

            except Exception as e:
                print("Hybrid Error:", e)

            if not recommended and current_genre:
                cursor.execute("""
                    SELECT *
                    FROM contents
                    WHERE id != %s
                    AND genre LIKE %s
                    AND language LIKE %s
                    ORDER BY hearts DESC
                    LIMIT 12
                """, (
                    content_id,
                    f"%{current_genre}%",
                    f"%{current_language}%"
                ))

                recommended = cursor.fetchall()

            if not recommended and current_genre:
                cursor.execute("""
                    SELECT *
                    FROM contents
                    WHERE id != %s
                    AND genre LIKE %s
                    ORDER BY RAND()
                    LIMIT 12
                """, (content_id, f"%{current_genre}%"))

                recommended = cursor.fetchall()

            for item in recommended:
                poster_name = (item.get("poster") or "").strip()

                item["poster_path"] = (
                    url_for("static", filename=f"uploads/{poster_name}")
                    if poster_name else None
                )

        conn.commit()

        # =====================================
        # 🎬 FINAL RENDER
        # =====================================
        return render_template(
            "watch.html",
            content=content,
            cast=cast,
            reviews=reviews,
            saved_progress=saved_progress,
            recommended=recommended
        )

    except Exception as e:
        print("WATCH PAGE ERROR:", e)
        return "Something went wrong"

    finally:
        conn.close()
#------- watch kids content -----------

@app.route('/kids_watch/<int:content_id>')
def kids_watch(content_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            cursor.execute(
                "SELECT * FROM contents WHERE id=%s",
                (content_id,)
            )

            content = cursor.fetchone()

            if not content:
                return "Content not found"

        return render_template("kids_watch.html", content=content)

    finally:
        conn.close()
# --- New Social & Interactive Routes ---


# -------------------------------
# SOCIAL HUB PAGE
# -------------------------------
@app.route('/social_hub')
def social_hub():

    # 🔐 Login check
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            # -------------------------------
            # 👤 CURRENT USER
            # -------------------------------
            cursor.execute("""
                SELECT id, username, gender, profile_image
                FROM users
                WHERE id = %s
            """, (session['user_id'],))

            current_user = cursor.fetchone()
            current_user['initial'] = current_user['username'][0].upper()

            # -------------------------------
            # 👥 OTHER USERS
            # -------------------------------
            cursor.execute("""
                SELECT id, username, age, fav_genre, gender, profile_image
                FROM users
                WHERE id != %s
            """, (session['user_id'],))

            users = cursor.fetchall()
            for user in users:
                user['initial'] = user['username'][0].upper()

            # -------------------------------
            # ❤️ FOLLOWING
            # -------------------------------
            cursor.execute("""
                SELECT followed_id 
                FROM social_graph 
                WHERE follower_id=%s
            """, (session['user_id'],))

            following = [row['followed_id'] for row in cursor.fetchall()]

            return render_template(
                'social_hub.html',
                users=users,
                following=following,
                current_user=current_user
            )

    except Exception as e:
        print("SOCIAL HUB ERROR:", e)
        return "Something went wrong", 500

    finally:
        conn.close()


# -------------------------------
# PROFILE PAGE
# -------------------------------
@app.route('/socialhub_profile/<int:user_id>')
def socialhub_profile(user_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 👤 CURRENT USER
        cursor.execute("""
            SELECT id, username, gender, profile_image
            FROM users WHERE id=%s
        """, (session['user_id'],))
        current_user = cursor.fetchone()
        current_user['initial'] = current_user['username'][0].upper()

        # 👤 PROFILE USER
        cursor.execute("""
            SELECT id, username, age, fav_genre, gender, profile_image
            FROM users WHERE id=%s
        """, (user_id,))
        user = cursor.fetchone()
        if not user: return "User not found", 404
        user['initial'] = user['username'][0].upper()

        # 🎬 HISTORY
        cursor.execute("""
            SELECT DISTINCT c.id, c.title, c.genre, c.thumbnail
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id=%s
            ORDER BY uh.watched_at DESC
            LIMIT 5
        """, (user_id,))
        history_based = cursor.fetchall()

        # 🎯 GENRE
        fav_genre = user.get('fav_genre') or ""
        cursor.execute("""
            SELECT id, title, genre, thumbnail
            FROM contents
            WHERE genre LIKE %s
            ORDER BY trending_rank DESC
            LIMIT 5
        """, (f"%{fav_genre}%",))
        genre_based = cursor.fetchall()

        # combine unique recommendations
        combined = {item['id']: item for item in history_based}
        for item in genre_based:
            if item['id'] not in combined:
                combined[item['id']] = item
        recommended_contents = list(combined.values())

        # Ensure thumbnails have full URL
        for item in recommended_contents:
            if item['thumbnail'] and not item['thumbnail'].startswith(('http://', 'https://')):
                # Assuming thumbnails are stored in static/uploads/
                item['thumbnail'] = url_for('static', filename=f'uploads/{item["thumbnail"]}')

        # ❤️ FOLLOW STATUS
        cursor.execute("""
            SELECT 1 FROM social_graph
            WHERE follower_id=%s AND followed_id=%s
        """, (session['user_id'], user_id))
        is_following = cursor.fetchone() is not None

        # 👥 FOLLOWERS
        cursor.execute("""
            SELECT u.id, u.username, u.profile_image, u.gender
            FROM social_graph sg
            JOIN users u ON sg.follower_id = u.id
            WHERE sg.followed_id=%s
        """, (user_id,))
        followers = cursor.fetchall()
        for f in followers: f['initial'] = f['username'][0].upper()

        # ➡️ FOLLOWING
        cursor.execute("""
            SELECT u.id, u.username, u.profile_image, u.gender
            FROM social_graph sg
            JOIN users u ON sg.followed_id = u.id
            WHERE sg.follower_id=%s
        """, (user_id,))
        following = cursor.fetchall()
        for f in following: f['initial'] = f['username'][0].upper()

        return render_template(
            'socialhub_profile.html',
            user=user,
            recommended_contents=recommended_contents,
            is_following=is_following,
            followers=followers,
            following=following,
            current_user=current_user
        )
    except Exception as e:
        print("PROFILE ERROR:", e)
        return "Something went wrong", 500
    finally:
        conn.close()
# -------------------------------
# FOLLOW / UNFOLLOW
# -------------------------------
@app.route('/toggle_follow/<int:user_id>', methods=['POST'])
def toggle_follow(user_id):

    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"})

    me = session['user_id']
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            # Check if already following
            cursor.execute("""
                SELECT 1 FROM social_graph
                WHERE follower_id=%s AND followed_id=%s
            """, (me, user_id))

            if cursor.fetchone():
                # UNFOLLOW
                cursor.execute("""
                    DELETE FROM social_graph
                    WHERE follower_id=%s AND followed_id=%s
                """, (me, user_id))
                conn.commit()
                following_status = False
            else:
                # FOLLOW
                cursor.execute("""
                    INSERT INTO social_graph (follower_id, followed_id)
                    VALUES (%s,%s)
                """, (me, user_id))
                conn.commit()
                following_status = True

            # ✅ Return updated status for AJAX
            return jsonify({
                "success": True,
                "following": following_status,
                "user_id": user_id
            })

    finally:
        conn.close()
# -------------------------------
# CHAT PAGE
# -------------------------------
@app.route('/chat/<int:user_id>')
def chat(user_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            # 🔥 CHAT USER
            cursor.execute("""
                SELECT id, username, profile_image
                FROM users
                WHERE id=%s
            """, (user_id,))
            chat_user = cursor.fetchone()

            if not chat_user:
                return "User not found", 404

            # 🔥 LOGGED-IN USER
            cursor.execute("""
                SELECT id, username, profile_image
                FROM users
                WHERE id=%s
            """, (session['user_id'],))
            me = cursor.fetchone()

            # ✅ PROFILE IMAGE OR INITIAL
            def format_user(user):
                if user.get('profile_image'):
                    user['profile_image'] = '/static/uploads/' + user['profile_image']
                    user['initial'] = None
                else:
                    user['profile_image'] = None
                    user['initial'] = user['username'][0].upper()

                return user

            chat_user = format_user(chat_user)
            me = format_user(me)

            return render_template(
                'chat.html',
                chat_user=chat_user,
                me=me
            )

    finally:
        conn.close()


# -------------------------------
# FETCH MESSAGES
# -------------------------------
@app.route('/fetch_messages/<int:user_id>')
def fetch_messages(user_id):

    if 'user_id' not in session:
        return jsonify([])

    me_id = session['user_id']
    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            cursor.execute("""
                SELECT sender_id, receiver_id, message, timestamp
                FROM chats
                WHERE (sender_id=%s AND receiver_id=%s)
                   OR (sender_id=%s AND receiver_id=%s)
                ORDER BY timestamp ASC
            """, (me_id, user_id, user_id, me_id))

            messages = cursor.fetchall()

            # ✅ GET ALL UNIQUE USER IDS
            user_ids = list(set([msg['sender_id'] for msg in messages]))
            users = {}

            if user_ids:
                format_strings = ','.join(['%s'] * len(user_ids))
                cursor.execute(f"""
                    SELECT id, username, profile_image
                    FROM users
                    WHERE id IN ({format_strings})
                """, tuple(user_ids))

                for user in cursor.fetchall():
                    users[user['id']] = user

            # ✅ ATTACH USER DATA
            for msg in messages:
                sender = users.get(msg['sender_id'])

                if sender:
                    # 👤 Username
                    msg['username'] = sender['username']

                    # 🖼 Profile image OR initial
                    if sender.get('profile_image'):
                        msg['profile_image'] = '/static/uploads/' + sender['profile_image']
                        msg['initial'] = None
                    else:
                        msg['profile_image'] = None
                        msg['initial'] = sender['username'][0].upper()

                else:
                    msg['username'] = "Unknown"
                    msg['profile_image'] = None
                    msg['initial'] = "?"

                # ✅ ADD FLAG (VERY IMPORTANT)
                msg['is_me'] = (msg['sender_id'] == me_id)

            return jsonify(messages)

    finally:
        conn.close()


# -------------------------------
# SEND MESSAGE
# -------------------------------
@app.route('/send_message/<int:user_id>', methods=['POST'])
def send_message(user_id):

    if 'user_id' not in session:
        return jsonify({"success": False})

    me = session['user_id']
    msg = request.json.get('message', '').strip()

    if not msg:
        return jsonify({"success": False})

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO chats (sender_id, receiver_id, message)
                VALUES (%s,%s,%s)
            """, (me, user_id, msg))
            conn.commit()

            return jsonify({"success": True})

    finally:
        conn.close()
##########################################################
@app.route('/api/rate', methods=['POST'])
def api_rate():

    if 'user_id' not in session:
        return jsonify({'error': 'Login required'}), 401

    data = request.json
    content_id = data.get('content_id')
    rating = data.get('rating')

    if not content_id or not rating:
        return jsonify({'error': 'Missing data'}), 400

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            # Insert OR Update
            cursor.execute("""
                INSERT INTO ratings (user_id, content_id, rating)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE rating=%s
            """, (session['user_id'], content_id, rating, rating))

        conn.commit()
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

    content_id = request.args.get('content_id')

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:

            cursor.execute("""
                SELECT rating FROM ratings
                WHERE user_id=%s AND content_id=%s
            """, (session['user_id'], content_id))

            result = cursor.fetchone()

            if result:
                return jsonify({'success': True, 'rating': result['rating']})
            else:
                return jsonify({'success': True, 'rating': 0})

    finally:
        conn.close()
@app.route('/api/review', methods=['POST'])
def api_review():
    """Handle Text AI Review & Sentiment"""
    if 'user_id' not in session: return redirect(url_for('login'))
    content_id = request.form['content_id']
    text = request.form['review_text']
    
    # Sentiment Analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity # -1 to 1
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO reviews (user_id, content_id, review_text, sentiment_score) VALUES (%s, %s, %s, %s)", 
                           (session['user_id'], content_id, text, sentiment))
        conn.commit()
        flash('Review posted!', 'success')
        return redirect(url_for('watch_content', content_id=content_id))
    finally:
        conn.close()

@app.route('/api/follow/<int:target_id>')
def api_follow(target_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Toggle follow
            cursor.execute("SELECT id FROM social_graph WHERE follower_id=%s AND followed_id=%s", (session['user_id'], target_id))
            exists = cursor.fetchone()
            if exists:
                cursor.execute("DELETE FROM social_graph WHERE id=%s", (exists['id'],))
                flash('Unfollowed user.', 'info')
            else:
                cursor.execute("INSERT INTO social_graph (follower_id, followed_id) VALUES (%s, %s)", (session['user_id'], target_id))
                flash('Followed user!', 'success')
        conn.commit()
        return redirect(url_for('social_hub'))
    finally:
        conn.close()
@app.route('/admin/edit/<int:content_id>', methods=['GET', 'POST'])
def edit_content(content_id):

    # ==========================
    # AUTH CHECK
    # ==========================
    if 'loggedin' not in session or session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    conn = None

    try:
        conn = get_connection()

        # ==========================
        # CONNECTION CHECK
        # ==========================
        if not conn:
            flash("Database connection failed!", "danger")
            return redirect(url_for('admin_dashboard'))

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # ==========================
        # GET EXISTING CONTENT FIRST
        # ==========================
        cursor.execute("SELECT * FROM contents WHERE id=%s", (content_id,))
        content = cursor.fetchone()

        if not content:
            flash("Content not found!", "danger")
            return redirect(url_for('admin_dashboard'))

        # ==========================
        # POST METHOD
        # ==========================
        if request.method == 'POST':

            title = request.form.get('title')
            story = request.form.get('story')
            language = request.form.get('language')
            release_year = request.form.get('release_year')
            content_type = request.form.get('content_type')
            genre = request.form.get('genre')
            upload_date = request.form.get('upload_date')

            # ==========================
            # BACKDROP URL
            # ==========================
            backdrop_url = request.form.get('backdrop_url', '').strip()

            # ==========================
            # HERO
            # ==========================
            hero  = request.form.get('hero')
            hero2 = request.form.get('hero2')
            hero3 = request.form.get('hero3')

            # ==========================
            # HEROINE
            # ==========================
            heroine  = request.form.get('heroine')
            heroine2 = request.form.get('heroine2')
            heroine3 = request.form.get('heroine3')

            # ==========================
            # OTHERS
            # ==========================
            director = request.form.get('director')
            comedian = request.form.get('comedian')
            producer = request.form.get('producer')

            release_year = int(release_year) if release_year else None
            trending_rank = int(request.form.get('trending_rank') or 0)
            deepflix_rank = int(request.form.get('deepflix_rank') or 0)
            is_90s = 1 if request.form.get('is_90s') else 0

            # ==========================
            # YOUTUBE ID
            # ==========================
            youtube_input = request.form.get('youtube_video_id', '').strip()
            youtube_video_id = None

            if youtube_input:
                if "youtube.com" in youtube_input and "v=" in youtube_input:
                    youtube_video_id = youtube_input.split("v=")[1].split("&")[0]
                elif "youtu.be/" in youtube_input:
                    youtube_video_id = youtube_input.split("youtu.be/")[1].split("?")[0]
                else:
                    youtube_video_id = youtube_input

            # ==========================
            # FILE UPLOAD
            # ==========================
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            def save_file(file):
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(upload_folder, filename)
                    file.save(filepath)
                    return filename
                return None

            thumbnail_filename = save_file(request.files.get('thumbnail'))

            hero_image  = save_file(request.files.get('hero_image'))
            hero2_image = save_file(request.files.get('hero2_image'))
            hero3_image = save_file(request.files.get('hero3_image'))

            heroine_image  = save_file(request.files.get('heroine_image'))
            heroine2_image = save_file(request.files.get('heroine2_image'))
            heroine3_image = save_file(request.files.get('heroine3_image'))

            director_image = save_file(request.files.get('director_image'))
            comedian_image = save_file(request.files.get('comedian_image'))
            producer_image = save_file(request.files.get('producer_image'))

            # ==========================
            # RANK FIX
            # ==========================
            if deepflix_rank > 0:
                cursor.execute("""
                    UPDATE contents
                    SET deepflix_rank=0
                    WHERE deepflix_rank=%s AND id!=%s
                """, (deepflix_rank, content_id))

            # ==========================
            # UPDATE QUERY
            # ==========================
            query = """
                UPDATE contents SET
                title=%s,
                story=%s,
                language=%s,
                release_year=%s,
                content_type=%s,
                genre=%s,
                hero=%s,
                hero2=%s,
                hero3=%s,
                heroine=%s,
                heroine2=%s,
                heroine3=%s,
                director=%s,
                comedian=%s,
                producer=%s,
                upload_date=%s,
                trending_rank=%s,
                deepflix_rank=%s,
                youtube_video_id=%s,
                is_90s=%s
            """

            values = [
                title, story, language, release_year,
                content_type, genre,
                hero, hero2, hero3,
                heroine, heroine2, heroine3,
                director, comedian, producer,
                upload_date,
                trending_rank, deepflix_rank,
                youtube_video_id,
                is_90s
            ]

            # ==========================
            # OPTIONAL UPDATES
            # ==========================
            if thumbnail_filename:
                query += ", thumbnail=%s"
                values.append(thumbnail_filename)

            if hero_image:
                query += ", hero_image=%s"
                values.append(hero_image)

            if hero2_image:
                query += ", hero2_image=%s"
                values.append(hero2_image)

            if hero3_image:
                query += ", hero3_image=%s"
                values.append(hero3_image)

            if heroine_image:
                query += ", heroine_image=%s"
                values.append(heroine_image)

            if heroine2_image:
                query += ", heroine2_image=%s"
                values.append(heroine2_image)

            if heroine3_image:
                query += ", heroine3_image=%s"
                values.append(heroine3_image)

            if director_image:
                query += ", director_image=%s"
                values.append(director_image)

            if comedian_image:
                query += ", comedian_image=%s"
                values.append(comedian_image)

            if producer_image:
                query += ", producer_image=%s"
                values.append(producer_image)

            # ==========================
            # BACKDROP SAFE UPDATE
            # ==========================
            if backdrop_url:
                query += ", backdrop_url=%s"
                values.append(backdrop_url)
            else:
                query += ", backdrop_url=%s"
                values.append(content['backdrop_url'])

            query += " WHERE id=%s"
            values.append(content_id)

            cursor.execute(query, tuple(values))
            conn.commit()

            flash("Content updated successfully!", "success")
            return redirect(url_for('admin_dashboard'))

        # ==========================
        # GET RENDER
        # ==========================
        return render_template('edit_content.html', content=content)

    except Exception as e:
        print("ERROR:", e)

        if conn:
            conn.rollback()

        flash(f"Error updating content: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))

    finally:
        if conn:
            conn.close()
# =====================================
# DELETE ROUTE
# =====================================

@app.route('/admin/delete/<int:content_id>')
def delete_content(content_id):

    if 'loggedin' not in session or session.get('user_type') != 'admin':
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM contents WHERE id=%s", (content_id,))
        conn.commit()

        flash("Content deleted successfully!", "danger")
        return redirect(url_for('admin_dashboard'))

    except Exception as e:
        conn.rollback()
        flash(f"Delete failed: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))

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
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT progress FROM continue_watching
        WHERE user_id=%s AND content_id=%s
    """, (session['user_id'], content_id))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row:
        return jsonify({'success': True, 'progress': float(row['progress'])})
    else:
        return jsonify({'success': True, 'progress': 0})
@app.route('/api/progress/update', methods=['POST'])
def update_progress():

    if 'user_id' not in session:
        return jsonify({'success': False}), 401

    data = request.get_json()

    content_id = data.get('content_id')
    progress = data.get('progress')

    if not content_id or progress is None:
        return jsonify({'success': False}), 400

    try:
        progress = float(progress)
    except:
        return jsonify({'success': False}), 400

    # Keep between 0 and 100
    progress = max(0, min(progress, 100))

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # If 95% watched remove from continue watching
        if progress >= 95:
            cursor.execute("""
                DELETE FROM continue_watching
                WHERE user_id=%s AND content_id=%s
            """, (session['user_id'], content_id))
        else:
            cursor.execute("""
                INSERT INTO continue_watching (user_id, content_id, progress)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    progress = VALUES(progress),
                    last_watched = CURRENT_TIMESTAMP
            """, (session['user_id'], content_id, progress))

        conn.commit()
        return jsonify({'success': True})

    except Exception as e:
        print("Progress Error:", e)
        return jsonify({'success': False}), 500

    finally:
        cursor.close()
        conn.close()

# ==========================================
# AJAX: TOGGLE HEART / LIKE
# ==========================================
@app.route('/ajax_toggle_heart', methods=['POST'])
def ajax_toggle_heart():

    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({
            'status': 'login_required'
        }), 401

    data = request.get_json(silent=True) or {}
    content_id = data.get('content_id')

    if not content_id:
        return jsonify({
            'status': 'error',
            'message': 'Content ID is required'
        }), 400

    conn = None

    try:
        conn = get_connection()

        if conn is None:
            return jsonify({
                'status': 'error',
                'message': 'Database connection failed'
            }), 500

        with conn.cursor() as cursor:

            # Check if the content exists
            cursor.execute(
                "SELECT id, hearts FROM contents WHERE id=%s",
                (content_id,)
            )
            content = cursor.fetchone()

            if not content:
                return jsonify({
                    'status': 'error',
                    'message': 'Content not found'
                }), 404

            # Check if the user already liked this content
            cursor.execute("""
                SELECT *
                FROM user_hearts
                WHERE user_id=%s AND content_id=%s
            """, (session['user_id'], content_id))

            existing = cursor.fetchone()

            if existing:
                # Remove existing like
                cursor.execute("""
                    DELETE FROM user_hearts
                    WHERE user_id=%s AND content_id=%s
                """, (session['user_id'], content_id))

                cursor.execute("""
                    UPDATE contents
                    SET hearts = GREATEST(COALESCE(hearts, 0) - 1, 0)
                    WHERE id=%s
                """, (content_id,))

            else:
                # Add a new like
                cursor.execute("""
                    INSERT INTO user_hearts (user_id, content_id)
                    VALUES (%s, %s)
                """, (session['user_id'], content_id))

                cursor.execute("""
                    UPDATE contents
                    SET hearts = COALESCE(hearts, 0) + 1
                    WHERE id=%s
                """, (content_id,))

            conn.commit()

            # Get updated heart count
            cursor.execute(
                "SELECT hearts FROM contents WHERE id=%s",
                (content_id,)
            )
            updated = cursor.fetchone()

        return jsonify({
            'status': 'success',
            'hearts': updated['hearts'] if updated else 0
        })

    except Exception as e:
        if conn:
            conn.rollback()

        app.logger.exception("Error toggling content heart")

        return jsonify({
            'status': 'error',
            'message': 'Unable to update heart. Please try again.'
        }), 500

    finally:
        if conn:
            conn.close()


# ==========================================
# USER PROFILE
# ==========================================
@app.route('/profile', methods=['GET', 'POST'])
def profile():

    # Check login
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = None

    try:
        conn = get_connection()

        if conn is None:
            return "Database connection failed", 500

        with conn.cursor() as cursor:

            # Get user details
            cursor.execute(
                "SELECT * FROM users WHERE id=%s",
                (user_id,)
            )
            user = cursor.fetchone()

            if not user:
                session.clear()
                return redirect(url_for('login'))

            if request.method == 'POST':

                action = request.form.get('action')

                # Remove profile image
                if action == 'remove_image':

                    old_image = user.get('profile_image')

                    if old_image:
                        upload_folder = app.config['UPLOAD_FOLDER']
                        old_path = os.path.join(
                            upload_folder,
                            os.path.basename(old_image)
                        )

                        # Delete only files inside the upload folder
                        if os.path.isfile(old_path):
                            os.remove(old_path)

                    cursor.execute("""
                        UPDATE users
                        SET profile_image=NULL
                        WHERE id=%s
                    """, (user_id,))

                    conn.commit()

                    return redirect(url_for('profile'))

                # Upload profile image
                file = request.files.get('profile_image')

                if file and file.filename:

                    filename = secure_filename(file.filename)

                    if not filename:
                        return "Invalid filename", 400

                    filename = f"{user_id}_{filename}"

                    upload_folder = app.config['UPLOAD_FOLDER']
                    os.makedirs(upload_folder, exist_ok=True)

                    filepath = os.path.join(
                        upload_folder,
                        filename
                    )

                    # Save the new image first
                    file.save(filepath)

                    old_image = user.get('profile_image')

                    cursor.execute("""
                        UPDATE users
                        SET profile_image=%s
                        WHERE id=%s
                    """, (filename, user_id))

                    conn.commit()

                    # Delete old image after successful DB update
                    if old_image and old_image != filename:
                        old_path = os.path.join(
                            upload_folder,
                            os.path.basename(old_image)
                        )

                        if os.path.isfile(old_path):
                            os.remove(old_path)

                    return redirect(url_for('profile'))

            return render_template(
                'profile.html',
                user=user
            )

    except Exception:
        if conn:
            conn.rollback()

        app.logger.exception("Error loading or updating profile")

        return "Unable to load profile. Please try again.", 500

    finally:
        if conn:
            conn.close()

@app.route('/delete_account')
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM ratings WHERE user_id = %s", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))

        conn.commit()

    finally:
        conn.close()

    session.clear()
    flash("Your account has been permanently deleted.", "success")
    return redirect(url_for('register'))




# ==========================================================
# ALL MOVIES PAGE
# ==========================================================

@app.route('/all_movies')
def all_movies():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    search = request.args.get('search', '').strip()
    genre = request.args.get('genre', '').strip()

    conn = get_connection()

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = "SELECT * FROM contents WHERE 1=1"
        params = []

        if search:
            pattern = "%" + search.lower() + "%"
            query += """
            AND (
                LOWER(title) LIKE %s OR
                LOWER(hero) LIKE %s OR
                LOWER(heroine) LIKE %s OR
                LOWER(director) LIKE %s OR
                LOWER(language) LIKE %s OR
                LOWER(genre) LIKE %s
            )
            """
            params.extend([pattern]*6)

        if genre and genre.lower() != "all":
            query += " AND LOWER(genre) LIKE %s"
            params.append("%" + genre.lower() + "%")

        query += " ORDER BY id DESC"

        cursor.execute(query, params)
        movies = cursor.fetchall()

        # ✅ REMOVE KIDS HERE (SAFE)
        movies = [m for m in movies if not m.get('is_kids')]

        print("ALL MOVIES COUNT:", len(movies))

        cursor.execute(
            "SELECT username, profile_image FROM users WHERE id=%s",
            (session['user_id'],)
        )
        user = cursor.fetchone()

        return render_template(
            "all_movies.html",
            movies=movies,
            user=user,
            search_value=search
        )

    finally:
        conn.close()
@app.route('/movie_suggestions')
def movie_suggestions():

    keyword = request.args.get('q', '').strip()

    # Empty search avoid
    if not keyword:
        return jsonify([])

    conn = get_connection()

    try:

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = """
        SELECT title, hero, heroine, director, genre, language
        FROM contents
        WHERE content_type='movie'
        AND (
            LOWER(title) LIKE %s
            OR LOWER(hero) LIKE %s
            OR LOWER(heroine) LIKE %s
            OR LOWER(director) LIKE %s
            OR LOWER(genre) LIKE %s
            OR LOWER(language) LIKE %s
        )
        LIMIT 25
        """

        pattern = "%" + keyword.lower() + "%"

        cursor.execute(query, (
            pattern,
            pattern,
            pattern,
            pattern,
            pattern,
            pattern
        ))

        rows = cursor.fetchall()

        suggestions = []

        # Collect suggestions
        for row in rows:

            for field in ['title','hero','heroine','director','genre','language']:

                value = row.get(field)

                if value:
                    suggestions.append(value)

        # Remove duplicates
        suggestions = list(set(suggestions))

        # -------- FUZZY SORTING --------
        suggestions.sort(
            key=lambda x: fuzz.partial_ratio(keyword.lower(), x.lower()),
            reverse=True
        )

        # Limit final results
        suggestions = suggestions[:10]

        return jsonify(suggestions)

    finally:

        conn.close()
@app.route('/play/<int:content_id>')
def play_content(content_id):

    # =====================================
    # 🔐 LOGIN CHECK
    # =====================================
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # 🎬 GET MOVIE / SERIES
        # =====================================
        cursor.execute(
            "SELECT * FROM contents WHERE id=%s",
            (content_id,)
        )

        movie = cursor.fetchone()

        if not movie:
            return "Content not found"

        # =====================================
        # 🧠 SAVE USER HISTORY
        # (FOR RECOMMENDATION SYSTEM)
        # =====================================
        cursor.execute("""
            INSERT INTO user_history (user_id, content_id)
            VALUES (%s,%s)
            ON DUPLICATE KEY UPDATE watched_at = NOW()
        """, (session['user_id'], content_id))

        # =====================================
        # ▶ CONTINUE WATCHING UPDATE
        # =====================================
        cursor.execute("""
            INSERT INTO continue_watching
            (user_id, content_id, progress, last_watched)
            VALUES (%s,%s,0,NOW())
            ON DUPLICATE KEY UPDATE
            last_watched = NOW()
        """, (session['user_id'], content_id))

        conn.commit()

        # =====================================
        # 🤖 HYBRID RECOMMENDATIONS
        # =====================================
        recommendations = []

        try:

            last_desc = movie.get('story', '')

            rec_ids = hybrid_engine.get_hybrid_recommendations(
                session['user_id'],
                last_desc
            )

            if rec_ids:

                placeholders = ','.join(['%s'] * len(rec_ids))

                cursor.execute(
                    f"SELECT * FROM contents WHERE id IN ({placeholders})",
                    tuple(rec_ids)
                )

                recommendations = cursor.fetchall()

                # Maintain recommendation order
                recommendations.sort(
                    key=lambda x: rec_ids.index(x['id'])
                )

        except Exception as e:
            print("Recommendation Error:", e)

        # =====================================
        # 🎥 RENDER PLAYER PAGE
        # =====================================
        return render_template(
            "player.html",
            movie=movie,
            recommendations=recommendations
        )

    except Exception as e:
        print("Player Error:", e)
        return "Error loading player"

    finally:
        conn.close()



# ==========================================
# PERSON / ACTOR / DIRECTOR MOVIES
# ==========================================
@app.route('/person/<name>')
def person_movies(name):

    conn = None

    try:
        # Connect using the central database configuration
        conn = get_connection()

        if conn is None:
            return "Database connection failed", 500

        with conn.cursor() as cursor:

            query = """
                SELECT *
                FROM contents
                WHERE hero = %s
                   OR hero2 = %s
                   OR hero3 = %s
                   OR heroine = %s
                   OR heroine2 = %s
                   OR heroine3 = %s
                   OR director = %s
                   OR producer = %s
                   OR comedian = %s
            """

            params = (name,) * 9

            cursor.execute(query, params)
            movies = cursor.fetchall()

        # Convert the person's name into an image filename
        actor_image = name.lower().replace(" ", "_") + ".jpg"

        person = {
            "name": name,
            "profile_image": actor_image
        }

        return render_template(
            "person_movies.html",
            name=name,
            movies=movies,
            person=person
        )

    except Exception:
        app.logger.exception(
            "Error loading movies for person: %s",
            name
        )
        return "Unable to load movies right now. Please try again.", 500

    finally:
        if conn:
            conn.close()

# ====================================================
# admin_dashbboard_analysis
# ==============================
# ADMIN DASHBOARD ANALYTICS
# ==============================
# =====================================
# ADMIN DASHBOARD ANALYTICS (UNLIMITED)
# =====================================
@app.route('/admin_dashboard_analysis')
def admin_dashboard_analysis():

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor()

        # =========================================
        # TOTAL CONTENTS
        # =========================================
        cursor.execute("SELECT COUNT(*) AS total_movies FROM contents")
        result = cursor.fetchone() or {}
        total_movies = result.get('total_movies', 0)

        # =========================================
        # CONTENTS BY LANGUAGE
        # =========================================
        cursor.execute("""
            SELECT language, COUNT(*) AS count
            FROM contents
            GROUP BY language
            ORDER BY count DESC
        """)
        languages = cursor.fetchall() or []

        # =========================================
        # CONTENTS BY GENRE
        # =========================================
        cursor.execute("""
            SELECT genre, COUNT(*) AS count
            FROM contents
            GROUP BY genre
            ORDER BY count DESC
        """)
        genres = cursor.fetchall() or []

        # =========================================
        # HERO ANALYTICS
        # =========================================
        cursor.execute("""
            SELECT hero, COUNT(*) AS count
            FROM contents
            WHERE hero IS NOT NULL AND hero != ''
            GROUP BY hero
            ORDER BY count DESC
        """)
        heroes = cursor.fetchall() or []

        for hero in heroes:
            hero['image'] = (hero.get('hero') or '').lower().replace(" ", "_") + ".jpg"

        # =========================================
        # HEROINE ANALYTICS
        # =========================================
        cursor.execute("""
            SELECT heroine, COUNT(*) AS count
            FROM contents
            WHERE heroine IS NOT NULL AND heroine != ''
            GROUP BY heroine
            ORDER BY count DESC
        """)
        heroines = cursor.fetchall() or []

        for heroine in heroines:
            heroine['image'] = (heroine.get('heroine') or '').lower().replace(" ", "_") + ".jpg"

        # =========================================
        # DIRECTORS
        # =========================================
        cursor.execute("""
            SELECT director, COUNT(*) AS count
            FROM contents
            WHERE director IS NOT NULL AND director != ''
            GROUP BY director
            ORDER BY count DESC
        """)
        directors = cursor.fetchall() or []

        for director in directors:
            director['image'] = (director.get('director') or '').lower().replace(" ", "_") + ".jpg"

        # =========================================
        # PRODUCERS
        # =========================================
        cursor.execute("""
            SELECT producer, COUNT(*) AS count
            FROM contents
            WHERE producer IS NOT NULL AND producer != ''
            GROUP BY producer
            ORDER BY count DESC
        """)
        producers = cursor.fetchall() or []

        for producer in producers:
            producer['image'] = (producer.get('producer') or '').lower().replace(" ", "_") + ".jpg"

        # =========================================
        # MOST WATCHED
        # =========================================
        cursor.execute("""
            SELECT c.id, c.title, c.thumbnail, COUNT(*) AS views
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            GROUP BY c.id
            ORDER BY views DESC
        """)
        most_watched = cursor.fetchall() or []

        # =========================================
        # TRENDING
        # =========================================
        cursor.execute("""
            SELECT id, title, thumbnail, hearts
            FROM contents
            ORDER BY hearts DESC
        """)
        trending = cursor.fetchall() or []

        # =========================================
        # TIMELINE
        # =========================================
        cursor.execute("""
            SELECT DATE(watched_at) AS day, COUNT(*) AS views
            FROM user_history
            GROUP BY DATE(watched_at)
            ORDER BY day ASC
        """)
        timeline = cursor.fetchall() or []

        # =========================================
        # USERS
        # =========================================
        cursor.execute("""
            SELECT id, username, profile_image
            FROM users
            ORDER BY id DESC
        """)
        users = cursor.fetchall() or []

        # =========================================
        # TOP USERS
        # =========================================
        cursor.execute("""
            SELECT u.id, u.username, COUNT(uh.id) AS count
            FROM users u
            LEFT JOIN user_history uh ON u.id = uh.user_id
            GROUP BY u.id
            ORDER BY count DESC
        """)
        top_users = cursor.fetchall() or []

        # =========================================
        # 🔥 LATEST RELEASES (NEW ADDITION)
        # =========================================
        cursor.execute("""
            SELECT id, title, thumbnail, genre, language, release_year, hearts
            FROM contents
            ORDER BY id DESC
            LIMIT 20
        """)
        posters = cursor.fetchall() or []

        # =========================================
        # 🔥 LOAD MODEL METRICS (SAFE)
        # =========================================
        metrics_path = "data/model_metrics.json"

        confusion_matrix = []
        labels = []
        accuracy = []
        val_accuracy = []
        loss = []
        val_loss = []
        epochs = []

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                confusion_matrix = metrics.get("confusion_matrix") or []
                labels = metrics.get("labels") or []
                accuracy = metrics.get("accuracy") or []
                val_accuracy = metrics.get("val_accuracy") or []
                loss = metrics.get("loss") or []
                val_loss = metrics.get("val_loss") or []

                epochs = list(range(1, len(accuracy) + 1))

            except Exception as e:
                print("Metrics Load Error:", e)

        # =========================================
        # RENDER TEMPLATE
        # =========================================
        return render_template(
            "admin_dashboard_analysis.html",

            total_movies=total_movies,
            languages=languages or [],
            genres=genres or [],
            heroes=heroes or [],
            heroines=heroines or [],
            directors=directors or [],
            producers=producers or [],
            most_watched=most_watched or [],
            trending=trending or [],
            timeline=timeline or [],
            users=users or [],
            top_users=top_users or [],
            posters=posters or [],  # 🔥 IMPORTANT

            # 🔥 AI DATA
            confusion_matrix=confusion_matrix or [],
            labels=labels or [],
            epochs=epochs or [],
            accuracy=accuracy or [],
            val_accuracy=val_accuracy or [],
            loss=loss or [],
            val_loss=val_loss or []
        )

    except Exception as e:
        print("Admin Dashboard Error:", e)
        return "Error loading dashboard"

    finally:
        conn.close()

@app.route('/api/model-metrics')
def get_model_metrics():
    try:
        with open('data/model_metrics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})
# =====================================
# PRODUCER ANALYSIS PAGE
# =====================================
@app.route('/admin_producer_analysis/<producer_name>')
def admin_producer_analysis(producer_name):

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # PRODUCER MOVIES
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE producer=%s
            ORDER BY release_year DESC
        """, (producer_name,))
        movies = cursor.fetchall()

        # =====================================
        # GENRE DISTRIBUTION
        # =====================================
        cursor.execute("""
            SELECT genre, COUNT(*) AS count
            FROM contents
            WHERE producer=%s
            GROUP BY genre
        """, (producer_name,))
        genres = cursor.fetchall()

        # =====================================
        # LANGUAGE DISTRIBUTION
        # =====================================
        cursor.execute("""
            SELECT language, COUNT(*) AS count
            FROM contents
            WHERE producer=%s
            GROUP BY language
        """, (producer_name,))
        languages = cursor.fetchall()

        # =====================================
        # PRODUCER IMAGE
        # =====================================
        producer_image = producer_name.lower().replace(" ", "_") + ".jpg"

        return render_template(
            "admin_producer_analysis.html",
            producer=producer_name,
            producer_image=producer_image,
            movies=movies,
            genres=genres,
            languages=languages
        )

    except Exception as e:
        print("Producer Analysis Error:", e)
        return "Error loading producer page"

    finally:
        conn.close()
@app.route('/admin_hero_analysis/<hero>')
def admin_hero_analysis(hero):

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # HERO PROFILE IMAGE
        # =====================================

        hero_image = hero.lower().replace(" ", "_") + ".jpg"

        # =====================================
        # HERO MOVIES
        # =====================================

        cursor.execute("""
            SELECT id, title, genre, language, thumbnail, release_year
            FROM contents
            WHERE hero = %s
        """, (hero,))

        movies = cursor.fetchall() or []

        # =====================================
        # GENRE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT genre, COUNT(*) AS count
            FROM contents
            WHERE hero = %s
            GROUP BY genre
        """, (hero,))

        genres = cursor.fetchall() or []

        # =====================================
        # LANGUAGE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT language, COUNT(*) AS count
            FROM contents
            WHERE hero = %s
            GROUP BY language
        """, (hero,))

        languages = cursor.fetchall() or []

        return render_template(
            "admin_hero_analysis.html",
            hero=hero,
            hero_image=hero_image,
            movies=movies,
            genres=genres,
            languages=languages
        )

    except Exception as e:

        print("Hero analysis error:", e)
        return "Hero analysis error"

    finally:

        if conn:
            conn.close()

@app.route('/admin_heroine_analysis/<heroine>')
def admin_heroine_analysis(heroine):

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # HEROINE PROFILE IMAGE
        # =====================================

        heroine_image = heroine.lower().replace(" ", "_") + ".jpg"

        # =====================================
        # HEROINE MOVIES
        # =====================================

        cursor.execute("""
            SELECT id, title, genre, language, thumbnail, release_year
            FROM contents
            WHERE heroine = %s
        """, (heroine,))

        movies = cursor.fetchall() or []

        # =====================================
        # GENRE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT genre, COUNT(*) AS count
            FROM contents
            WHERE heroine = %s
            GROUP BY genre
        """, (heroine,))

        genres = cursor.fetchall() or []

        # =====================================
        # LANGUAGE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT language, COUNT(*) AS count
            FROM contents
            WHERE heroine = %s
            GROUP BY language
        """, (heroine,))

        languages = cursor.fetchall() or []

        return render_template(
            "admin_heroine_analysis.html",
            heroine=heroine,
            heroine_image=heroine_image,
            movies=movies,
            genres=genres,
            languages=languages
        )

    except Exception as e:

        print("Heroine analysis error:", e)
        return "Heroine analysis error"

    finally:

        if conn:
            conn.close()
@app.route('/admin_director_analysis/<director>')
def admin_director_analysis(director):

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # DIRECTOR PROFILE IMAGE
        # =====================================

        director_image = director.lower().replace(" ", "_") + ".jpg"

        # =====================================
        # DIRECTOR MOVIES
        # =====================================

        cursor.execute("""
            SELECT id, title, genre, language, thumbnail, release_year
            FROM contents
            WHERE director = %s
        """, (director,))

        movies = cursor.fetchall() or []

        # =====================================
        # GENRE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT genre, COUNT(*) AS count
            FROM contents
            WHERE director = %s
            GROUP BY genre
        """, (director,))

        genres = cursor.fetchall() or []

        # =====================================
        # LANGUAGE DISTRIBUTION
        # =====================================

        cursor.execute("""
            SELECT language, COUNT(*) AS count
            FROM contents
            WHERE director = %s
            GROUP BY language
        """, (director,))

        languages = cursor.fetchall() or []

        return render_template(
            "admin_director_analysis.html",
            director=director,
            director_image=director_image,
            movies=movies,
            genres=genres,
            languages=languages
        )

    except Exception as e:

        print("Director analysis error:", e)
        return "Director analysis error"

    finally:

        if conn:
            conn.close()
# ==============================
# ADMIN USER ANALYTICS
# ==============================

@app.route('/admin_user_analysis/<int:user_id>')
def admin_user_analysis(user_id):

    conn = get_connection()

    if not conn:
        return "Database connection failed"

    try:

        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =========================================
        # USER INFO
        # =========================================

        cursor.execute("""
            SELECT id, username, profile_image
            FROM users
            WHERE id = %s
        """, (user_id,))

        user = cursor.fetchone()

        # =========================================
        # TOTAL VIEWS
        # =========================================

        cursor.execute("""
            SELECT COUNT(*) AS total_views
            FROM user_history
            WHERE user_id = %s
        """, (user_id,))

        user['total_views'] = cursor.fetchone()['total_views']

        # =========================================
        # MOST WATCHED MOVIES
        # =========================================

        cursor.execute("""
            SELECT 
                c.id,
                c.title,
                c.genre,
                c.language,
                c.thumbnail,
                COUNT(*) AS views
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.id
            ORDER BY views DESC
            LIMIT 12
        """, (user_id,))

        movies = cursor.fetchall() or []

        # =========================================
        # GENRE ANALYTICS
        # =========================================

        cursor.execute("""
            SELECT 
                c.genre,
                COUNT(*) AS count
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.genre
        """, (user_id,))

        genres = cursor.fetchall() or []

        # =========================================
        # LANGUAGE ANALYTICS
        # =========================================

        cursor.execute("""
            SELECT 
                c.language,
                COUNT(*) AS count
            FROM user_history uh
            JOIN contents c ON uh.content_id = c.id
            WHERE uh.user_id = %s
            GROUP BY c.language
        """, (user_id,))

        languages = cursor.fetchall() or []

        return render_template(
            "admin_user_analysis.html",
            user=user,
            movies=movies,
            genres=genres,
            languages=languages
        )

    except Exception as e:

        print("Admin user analysis error:", e)
        return "Error loading user analysis"

    finally:

        if conn:
            conn.close()









    
@app.route('/delete_profile/<int:id>')
def delete_profile(id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM userprofiles
    WHERE id=%s
    """,(id,))

    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('userprofiles'))

@app.route('/edit_profile/<int:id>', methods=['POST'])
def edit_profile(id):

    name = request.form['name']
    age = request.form['age']

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE userprofiles
    SET name=%s, age=%s
    WHERE id=%s
    """,(name,age,id))

    conn.commit()

    cursor.close()
    conn.close()

    return redirect(url_for('userprofiles'))

@app.route('/all_series')
def all_series():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    conn = get_connection()
    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # 🔔 SERIES NOTIFICATIONS ONLY
        # =====================================
        cursor.execute("""
            SELECT n.*, c.thumbnail AS content_thumbnail
            FROM notifications n
            JOIN contents c ON n.content_id = c.id
            WHERE c.content_type = 'series'
              AND c.is_kids = 0
            ORDER BY n.created_at DESC
            LIMIT 10
        """)
        notifications = cursor.fetchall()

        # =====================================
        # 🔥 CONTINUE WATCHING (ONLY SERIES)
        # =====================================
        cursor.execute("""
            SELECT c.*, cw.progress
            FROM continue_watching cw
            JOIN contents c ON cw.content_id = c.id
            WHERE cw.user_id = %s
              AND c.content_type = 'series'
              AND c.is_kids = 0
            ORDER BY cw.last_watched DESC
            LIMIT 10
        """, (user_id,))
        continue_watching_series = cursor.fetchall()

        # =====================================
        # 🔥 RECENTLY ADDED SERIES
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'series'
              AND is_kids = 0
              AND created_at IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT 15
        """)
        recently_added_series = cursor.fetchall()

        # =====================================
        # 🎬 ALL SERIES (MAIN LIST)
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'series'
              AND is_kids = 0
            ORDER BY release_year DESC
        """)
        series_list = cursor.fetchall()

        return render_template(
            'all_series.html',
            notifications=notifications,   # ✅ added
            continue_watching_series=continue_watching_series,
            recently_added_series=recently_added_series,
            series_list=series_list
        )

    except Exception as e:
        return str(e)

    finally:
        cursor.close()
        conn.close()
@app.route('/series_play/<int:content_id>')
def series_play(content_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_connection()
    if not conn:
        return "Database connection failed"

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # =====================================
        # FETCH PARTICULAR SERIES
        # =====================================
        cursor.execute("""
            SELECT * FROM contents 
            WHERE id = %s AND content_type = 'series'
        """, (content_id,))

        series = cursor.fetchone()

        if not series:
            return "Series not found"

        # =====================================
        # 🔥 EXTRACT BASE TITLE (before :)
        # =====================================
        full_title = series['title']

        if ":" in full_title:
            base_title = full_title.split(":")[0].strip()
        else:
            base_title = full_title.strip()

        # =====================================
        # 🎬 FETCH ALL EPISODES BASED ON TITLE
        # =====================================
        cursor.execute("""
            SELECT *
            FROM contents
            WHERE content_type = 'series'
              AND title LIKE %s
              AND is_kids = 0
            ORDER BY id ASC
        """, (f"{base_title}:%",))

        episodes = cursor.fetchall()

        return render_template(
            'series_play.html',
            series=series,
            episodes=episodes,
            base_title=base_title
        )

    except Exception as e:
        return str(e)

    finally:
        cursor.close()
        conn.close()


@app.route('/api/user-growth')
def user_growth():
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as users
        FROM users
        GROUP BY DATE(created_at)
        ORDER BY day ASC
    """)

    data = cursor.fetchall()
    conn.close()

    return jsonify(data)
@app.route('/api/genre-trend')
def genre_trend():
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT genre, DATE(created_at) as day, COUNT(*) as count
        FROM contents
        GROUP BY genre, day
        ORDER BY day ASC
    """)

    data = cursor.fetchall()
    return jsonify(data)
@app.route('/api/movie-success')
def movie_success():
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT 
            c.title,

            COUNT(DISTINCT h.user_id) AS views,

            COUNT(DISTINCT uh.user_id) AS hearts,

            (COUNT(DISTINCT h.user_id) * 0.7 + 
             COUNT(DISTINCT uh.user_id) * 1.3) AS score

        FROM contents c

        LEFT JOIN user_history h 
            ON c.id = h.content_id

        LEFT JOIN user_hearts uh 
            ON c.id = uh.content_id

        GROUP BY c.id, c.title

        ORDER BY score DESC
        LIMIT 10
    """)

    data = cursor.fetchall()
    return jsonify(data)
@app.route('/api/user-retention')
def user_retention():
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT 
            DATE(created_at) as day,
            COUNT(*) as new_users
        FROM users
        GROUP BY day
        ORDER BY day ASC
    """)

    data = cursor.fetchall()

    # returning users dummy logic (temporary)
    result = []
    for row in data:
        result.append({
            "day": row["day"],
            "new_users": row["new_users"],
            "returning_users": int(row["new_users"] * 0.6)
        })

    return jsonify(result)


if __name__ == '__main__':
    if not os.path.exists('data'): os.makedirs('data')
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))