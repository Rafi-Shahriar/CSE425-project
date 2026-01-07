import os
import requests
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time

def download_file_resumable(url, dest_path, max_retries=5):
    """Download a file with resume capability and retries"""
    dest_path = Path(dest_path)
    temp_path = dest_path.with_suffix(dest_path.suffix + '.part')
    
    # Check if we have a partial download
    if temp_path.exists():
        resume_byte_pos = temp_path.stat().st_size
        print(f"Resuming download from {resume_byte_pos / (1024*1024):.1f} MB")
    else:
        resume_byte_pos = 0
    
    headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos > 0 else {}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Get total size
            if resume_byte_pos > 0:
                total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos
            else:
                total_size = int(response.headers.get('content-length', 0))
            
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            
            with open(temp_path, mode) as f, tqdm(
                desc=dest_path.name,
                initial=resume_byte_pos,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            # Download complete, rename temp file
            temp_path.rename(dest_path)
            print(f"✓ Successfully downloaded {dest_path.name}")
            return True
            
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                Exception) as e:
            print(f"\n⚠ Download interrupted (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                # Update resume position
                if temp_path.exists():
                    resume_byte_pos = temp_path.stat().st_size
                    headers = {'Range': f'bytes={resume_byte_pos}-'}
            else:
                print(f"✗ Failed to download after {max_retries} attempts")
                return False
    
    return False

def setup_fma_dataset(base_dir='./data'):
    """
    Download and setup FMA-small dataset
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    print("Setting up FMA-small dataset...")
    print("="*60)
    
    # FMA URLs
    fma_small_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    fma_metadata_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    
    # Download metadata first (smaller file, ~358MB)
    metadata_zip = base_path / "fma_metadata.zip"
    if not (base_path / "fma_metadata").exists():
        print("\n1. Downloading metadata (~358MB)...")
        print("This may take 10-30 minutes depending on your connection.")
        
        success = download_file_resumable(fma_metadata_url, metadata_zip)
        if not success:
            print("\n❌ Metadata download failed. Please try again.")
            print("The download will resume from where it left off.")
            return None
        
        print("\nExtracting metadata...")
        with zipfile.ZipFile(metadata_zip, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        print("✓ Metadata extracted")
        
        # Clean up zip file
        metadata_zip.unlink()
    else:
        print("✓ Metadata already exists")
    
    # Download audio files
    audio_zip = base_path / "fma_small.zip"
    if not (base_path / "fma_small").exists():
        print("\n2. Downloading FMA-small audio (~7.2GB)...")
        print("This will take 30-90 minutes depending on your connection.")
        print("If interrupted, just run the script again - it will resume!")
        
        success = download_file_resumable(fma_small_url, audio_zip)
        if not success:
            print("\n❌ Audio download failed. Please run the script again.")
            print("The download will resume from where it left off.")
            return None
        
        print("\nExtracting audio files (this may take 10-15 minutes)...")
        with zipfile.ZipFile(audio_zip, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        print("✓ Audio files extracted")
        
        # Clean up zip file
        audio_zip.unlink()
    else:
        print("✓ Audio files already exist")
    
    # Verify and load metadata
    print("\n3. Verifying dataset...")
    try:
        tracks = pd.read_csv(base_path / 'fma_metadata' / 'tracks.csv', 
                             index_col=0, header=[0, 1])
        genres = pd.read_csv(base_path / 'fma_metadata' / 'genres.csv', 
                             index_col=0)
        
        # Filter for small subset
        small = tracks['set', 'subset'] == 'small'
        tracks_small = tracks[small]
        
        print(f"\n{'='*60}")
        print("✓ Dataset ready!")
        print(f"{'='*60}")
        print(f"Total tracks in FMA-small: {len(tracks_small)}")
        print(f"Audio directory: {base_path / 'fma_small'}")
        print(f"Metadata directory: {base_path / 'fma_metadata'}")
        
        # Show genre distribution
        print(f"\nTop genres in dataset:")
        genre_counts = tracks_small['track', 'genre_top'].value_counts().head()
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} tracks")
        
        return base_path
        
    except Exception as e:
        print(f"\n⚠ Error loading metadata: {e}")
        return None

if __name__ == "__main__":
    print("FMA Dataset Downloader")
    print("This script will download the FMA-small dataset.")
    print("Total download size: ~7.6GB")
    print("Downloads are resumable - you can safely interrupt and restart!\n")
    
    data_path = setup_fma_dataset()
    
    if data_path:
        print("\n" + "="*60)
        print("✓ Setup complete! You can now start working on your project.")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the feature extraction script")
        print("2. Start implementing your VAE")
    else:
        print("\n" + "="*60)
        print("Setup incomplete. Please run the script again.")
        print("="*60)