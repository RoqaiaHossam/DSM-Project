import requests
import pandas as pd
import time
from datetime import datetime

api_key = "5822f6917661633710ff78b53d565405"
base_url = "https://api.themoviedb.org/3"

num_movies = 100
pages = 5
all_movies = []

for page in range(1, pages+1):
        url = f"{base_url}/movie/popular"
        params = {
            'api_key': api_key,
            'page': page,
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        movies = response.json()['results']

        all_movies.extend(movies)
        print(f"fetching page {page} , collected {len(movies)} movies")
        time.sleep(0.5)

print("=" * 50)
print(f"\n Total movies collected: {len(all_movies)}")

print("\nGetting movie details...")
movie_data = []

for i, movie in enumerate(all_movies[:num_movies]):
    movie_id = movie['id']
    
    url = f"{base_url}/movie/{movie_id}"
    params = {'api_key': api_key}
    
    response = requests.get(url, params=params)
    details = response.json()
    genres = ', '.join([g['name'] for g in details.get('genres', [])])
    movie_data.append({
        'title': details.get('title'),
        'release_date': details.get('release_date'),
        'runtime': details.get('runtime'),
        'budget': details.get('budget'),
        'revenue': details.get('revenue'),
        'rating': details.get('vote_average'),
        'votes': details.get('vote_count'),
        'popularity': details.get('popularity'),
        'language': details.get('original_language'),
        'genres': genres
    })

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1} movies")
    
    time.sleep(0.3)

df = pd.DataFrame(movie_data)
df.to_csv("movies_data.csv", index=False)
print(f"Done! Saved{len(df)} movies to movies_data.csv")
