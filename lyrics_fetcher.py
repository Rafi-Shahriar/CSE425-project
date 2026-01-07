"""
Fetch lyrics for FMA tracks using Genius API
You need to get a free API key from: https://genius.com/api-clients
"""

import lyricsgenius
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import time

class LyricsFetcher:
    """Fetch lyrics using Genius API"""
    
    def __init__(self, api_key):
        """
        Args:
            api_key: Genius API key (get from https://genius.com/api-clients)
        """
        self.genius = lyricsgenius.Genius(
            api_key,
            verbose=False,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"]
        )
        
    def fetch_lyrics(self, title, artist, max_retries=2):
        """
        Fetch lyrics for a song
        
        Args:
            title: Song title
            artist: Artist name
            max_retries: Number of retry attempts
            
        Returns:
            dict with lyrics and metadata
        """
        for attempt in range(max_retries):
            try:
                song = self.genius.search_song(title, artist)
                
                if song:
                    return {
                        'lyrics': song.lyrics,
                        'title': song.title,
                        'artist': song.artist,
                        'success': True,
                        'url': song.url
                    }
                else:
                    return {'success': False, 'reason': 'not_found'}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {'success': False, 'reason': str(e)}
        
        return {'success': False, 'reason': 'max_retries_exceeded'}
    
    def fetch_for_dataset(self, tracks_df, max_tracks=None, save_path='./data/lyrics.json'):
        """
        Fetch lyrics for FMA tracks
        
        Args:
            tracks_df: FMA tracks DataFrame
            max_tracks: Maximum number of tracks to process (None = all)
            save_path: Where to save lyrics
            
        Returns:
            dict of track_id -> lyrics data
        """
        lyrics_data = {}
        
        # Limit tracks if specified
        if max_tracks:
            tracks_df = tracks_df.head(max_tracks)
        
        print(f"Fetching lyrics for {len(tracks_df)} tracks...")
        print("This will take ~1 hour for 1000 tracks (API rate limits)")
        
        success_count = 0
        
        for idx, row in tqdm(tracks_df.iterrows(), total=len(tracks_df)):
            title = str(row[('track', 'title')])
            artist = str(row[('artist', 'name')])
            
            # Skip if title or artist is missing
            if title == 'nan' or artist == 'nan':
                continue
            
            # Fetch lyrics
            result = self.fetch_lyrics(title, artist)
            
            if result['success']:
                success_count += 1
                lyrics_data[str(idx)] = result
                
                # Save periodically (every 50 tracks)
                if success_count % 50 == 0:
                    self._save_lyrics(lyrics_data, save_path)
                    print(f"  Progress: {success_count} lyrics found")
            
            # Rate limiting (Genius API allows ~1 request per second)
            time.sleep(1.1)
        
        # Final save
        self._save_lyrics(lyrics_data, save_path)
        
        print(f"\n✓ Fetched lyrics for {success_count}/{len(tracks_df)} tracks")
        print(f"Success rate: {success_count/len(tracks_df)*100:.1f}%")
        
        return lyrics_data
    
    def _save_lyrics(self, lyrics_data, save_path):
        """Save lyrics to JSON file"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(lyrics_data, f, indent=2, ensure_ascii=False)


def load_lyrics(path='./data/lyrics.json'):
    """Load saved lyrics"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_api_key_from_user():
    """Interactive prompt for API key"""
    print("\n" + "="*80)
    print("GENIUS API KEY REQUIRED")
    print("="*80)
    print("\nTo fetch lyrics, you need a free Genius API key:")
    print("1. Go to: https://genius.com/api-clients")
    print("2. Sign up / Log in")
    print("3. Create a new API client")
    print("4. Copy the 'Client Access Token'")
    print()
    
    api_key = input("Paste your Genius API key here: ").strip()
    
    if not api_key:
        print("⚠ No API key provided. Using demo mode (no lyrics will be fetched)")
        return None
    
    # Save for future use
    with open('.genius_api_key', 'w') as f:
        f.write(api_key)
    
    return api_key


def load_saved_api_key():
    """Load previously saved API key"""
    key_file = Path('.genius_api_key')
    if key_file.exists():
        return key_file.read_text().strip()
    return None


if __name__ == "__main__":
    print("="*80)
    print("LYRICS FETCHER FOR FMA DATASET")
    print("="*80)
    
    # Load FMA metadata
    print("\nLoading FMA metadata...")
    tracks_file = './data/fma_metadata/tracks.csv'
    tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
    
    # Filter for small subset
    small = tracks['set', 'subset'] == 'small'
    tracks_small = tracks[small]
    
    # Filter for specific genres
    genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
    genre_mask = tracks_small['track', 'genre_top'].isin(genres)
    tracks_filtered = tracks_small[genre_mask]
    
    print(f"Total tracks to process: {len(tracks_filtered)}")
    
    # Get API key
    api_key = load_saved_api_key()
    if not api_key:
        api_key = get_api_key_from_user()
    
    if not api_key:
        print("\n⚠ No API key - exiting")
        print("Run this script again after getting your Genius API key")
        exit(1)
    
    # Ask user how many tracks to fetch
    print("\nHow many tracks to fetch lyrics for?")
    print("  - Full dataset: 3000 tracks (~2 hours)")
    print("  - Quick test: 100 tracks (~5 minutes)")
    print("  - Recommended: 1000 tracks (~1 hour)")
    
    max_tracks = input("\nEnter number [1000]: ").strip()
    max_tracks = int(max_tracks) if max_tracks else 1000
    
    # Initialize fetcher
    print("\nInitializing Genius API...")
    fetcher = LyricsFetcher(api_key)
    
    # Fetch lyrics
    lyrics_data = fetcher.fetch_for_dataset(
        tracks_filtered,
        max_tracks=max_tracks,
        save_path='./data/lyrics.json'
    )
    
    print("\n" + "="*80)
    print("✓ LYRICS FETCHING COMPLETE")
    print("="*80)
    print(f"Lyrics saved to: ./data/lyrics.json")
    print(f"Total lyrics fetched: {len(lyrics_data)}")
    print("\nNext step: Run text_features.py to extract text embeddings")