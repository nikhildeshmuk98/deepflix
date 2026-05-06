-- database.sql
-- Use this script to create the database and tables for the OTT recommendation system.

-- First, create the database
CREATE DATABASE IF NOT EXISTS ott_recommendation_db1;
USE ott_recommendation_db1;

-- Table for Admin credentials
CREATE TABLE IF NOT EXISTS admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Table for User registration details
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    fav_genre VARCHAR(100),
    fav_language VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for all the movie and series content
CREATE TABLE IF NOT EXISTS contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    story TEXT,
    language VARCHAR(50),
    release_year INT,
    content_type ENUM('movie', 'series') NOT NULL,
    genre VARCHAR(100),
    youtube_video_id VARCHAR(255),
    thumbnail VARCHAR(255),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store user's watching history
CREATE TABLE IF NOT EXISTS user_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    content_id INT,
    watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (content_id) REFERENCES contents(id)
);

-- Insert a default admin user. Password is 'admin123'.
-- The password 'admin123' is hashed using a function similar to `generate_password_hash`.
-- In Python, you can use `generate_password_hash('admin123')` to get the hashed value.
-- For now, let's insert a known hash.
-- Hashed value for 'admin123' using werkzeug.security.generate_password_hash:
-- 'sha256$r00t$4c000e84b8e5c1435f3378b276b539c3a373188d4c9d57a3e7e00843336d3969'
INSERT INTO admins (username, password) VALUES ('admin', 'sha256$28o81X94Vn0j571g$d9e4878a87607730e71954316a7f0506307122a275217983c270d4f59e99a80e');
