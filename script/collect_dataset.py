import lyricsgenius
import pandas as pd
import time

# Paste your Genius Access Token here
token = "zLhNJvK2ktK1yf4QH8vP6hAfHGBpFl20UETZxjxsQmeaa90kRNUgik9vo8lIJEHl"
genius = lyricsgenius.Genius(token)
genius.timeout = 15  # Increased from default 5 to 15 seconds
genius.retries = 3

# List of Retro/Classic artists
retro_artists = ["ABBA", "Bee Gees", "Queen", "Boney M.", "The Beatles"]

data = []
songs_per_artist = 500 

for artist_name in retro_artists:
    print(f"Searching for songs by {artist_name}...")
    try:
        # Search for the artist
        artist = genius.search_artist(artist_name, max_songs=songs_per_artist, sort="popularity")
        
        if artist:
            for song in artist.songs:
                # We use .get() to avoid errors if the key is missing
                song_data = song._body 
                
                # 2. Extract the year from release_date_components
                res_comp = song_data.get('release_date_components')
                
                if res_comp and 'year' in res_comp:
                    clean_year = str(res_comp['year'])
                else:
                    # Backup: Try to find a year in the 'release_date' string if it exists
                    import re
                    release_str = song_data.get('release_date') or ""
                    match = re.search(r'\d{4}', str(release_str))
                    clean_year = match.group(0) if match else "1980"

                data.append({
                    "artist_name": song.artist,
                    "track_name": song.title,
                    "release_date": clean_year,
                    "genre": "Retro",
                    "lyrics": song.lyrics
                })
        
        # Small sleep to avoid hitting API rate limits too hard
        time.sleep(1)

    except Exception as e:
        print(f"Error fetching {artist_name}: {e}")

# 2. Check if we actually got data before creating the DataFrame
if not data:
    print("CRITICAL ERROR: No data was collected. Check your internet or API token.")
else:
    # Create DataFrame
    student_dataset = pd.DataFrame(data)

    # Final Clean: Ensure year is exactly 4 digits (Requirement)
    # This prevents the KeyError by checking if column exists
    if 'release_date' in student_dataset.columns:
        student_dataset['release_date'] = student_dataset['release_date'].astype(str).str.extract(r'(\d{4})').fillna("1980")

    # Save to CSV
    student_dataset.to_csv("data/Student_dataset.csv", index=False)
    print(f"Success! Collected {len(student_dataset)} Retro songs.")