import pymysql.cursors

# Database Configuration
# Ensure these match your WAMP/XAMPP settings
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'ott_recommendation_db1', 
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def fix_database():
    print("🔌 Connecting to database...")
    try:
        connection = pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Please ensure WAMP/XAMPP MySQL is running and the database 'ott_recommendation_db1' exists.")
        return

    try:
        with connection.cursor() as cursor:
            print("🔍 Checking 'users' table schema...")
            
            # 1. Check and Add 'age'
            cursor.execute("SHOW COLUMNS FROM users LIKE 'age'")
            if not cursor.fetchone():
                print("   ➕ Adding missing column: 'age'...")
                cursor.execute("ALTER TABLE users ADD COLUMN age INT DEFAULT 25")
            else:
                print("   ✅ 'age' column already exists.")

            # 2. Check and Add 'gender'
            cursor.execute("SHOW COLUMNS FROM users LIKE 'gender'")
            if not cursor.fetchone():
                print("   ➕ Adding missing column: 'gender'...")
                cursor.execute("ALTER TABLE users ADD COLUMN gender VARCHAR(20) DEFAULT 'Other'")
            else:
                print("   ✅ 'gender' column already exists.")

            # 3. Create other missing tables for Hybrid Features
            print("🔨 Verifying Hybrid System tables...")
            
            # Ratings (ALS)
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
            
            # Reviews (Sentiment)
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

            # Social Graph (Social)
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

            # Likes (Implicit)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS likes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    content_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
        connection.commit()
        print("\n✅ Database Fixed Successfully!")
        print("You can now run 'python app.py' and register without errors.")

    except Exception as e:
        print(f"\n❌ Error updating database: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    fix_database()