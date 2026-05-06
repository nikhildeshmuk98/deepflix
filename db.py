import requests
import time
import csv

BEARER_TOKEN = "<REMOVED_TOKEN>"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

def fetch_top_movies(count=100, max_retries=3):
    movies = []
    page = 1

    while len(movies) < count:
        url = f"https://api.themoviedb.org/3/movie/top_rated?page={page}"
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                for movie in data["results"]:
                    movies.append(movie)
                    if len(movies) == count:
                        break
                print(f"✅ Page {page} fetched ({len(movies)} movies total)")
                page += 1
                time.sleep(0.5)
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"⚠️ Retry {retries}/{max_retries} for page {page} due to error: {e}")
                time.sleep(2)
        else:
            print(f"❌ Failed to fetch page {page} after {max_retries} retries.")
            break

    return movies

def save_movies_to_csv(movies, filename="top_100_movies.csv"):
    with open(filename, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "id", "title", "story", "language", "release_year", 
            "content_type", "genre", "youtube_video_id", "thumbnail"
        ])
        
        for m in movies:
            movie_id = m.get("id")
            title = m.get("title")
            overview = m.get("overview")
            language = m.get("original_language")
            release_year = m.get("release_date", "")[:4]
            content_type = "movie"
            genres = " / ".join(str(genre_id) for genre_id in m.get("genre_ids", []))
            youtube_video_id = ""  # TMDb API v3 doesn't include YouTube trailer directly here
            thumbnail = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else ""
            
            writer.writerow([
                movie_id, title, overview, language, release_year,
                content_type, genres, youtube_video_id, thumbnail
            ])
    print(f"📁 Saved to {filename}")

# Run the script
if __name__ == "__main__":
    movies = fetch_top_movies(count=100)
    save_movies_to_csv(movies)
