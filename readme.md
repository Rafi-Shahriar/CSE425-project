# VAE-Based Hybrid Music Clustering Project

## Project Overview
This project implements a Variational Autoencoder (VAE) based clustering pipeline for hybrid language music tracks. We explore different VAE architectures (Basic VAE, ConvVAE, Multimodal VAE, Beta-VAE) and evaluate their performance on multi-genre music clustering using both audio and text features.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Easy Task Files](#easy-task-files-20-marks)
- [Medium Task Files](#medium-task-files-25-marks)
- [Hard Task Files](#hard-task-files-25-marks)
- [Helper Scripts](#helper-scripts)
- [Execution Guide](#execution-guide)
- [Results and Outputs](#results-and-outputs)
- [Project Structure](#project-structure)

---

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- librosa, scikit-learn, pandas, numpy
- matplotlib, seaborn, umap-learn
- sentence-transformers (optional, for text features)
- lyricsgenius (for lyrics fetching)

---

## Dataset

**FMA-small Dataset**
- 8,000 tracks from Free Music Archive
- 5 genres used: Hip-Hop, Pop, Folk, Experimental, Rock
- 600 tracks per genre = 3,000 total tracks
- ~500 tracks with lyrics (via Genius API)
- ~2,500 tracks use metadata as text fallback

---

## Easy Task Files

### Core Implementation Files

#### 1. `dataset.py`
**Purpose:** Audio feature extraction and preprocessing  
**What it does:**
- Loads FMA-small metadata
- Extracts MFCC features from audio files (40D: 20 mean + 20 std)
- Normalizes features
- Saves processed features

**Execution:**
```bash
python dataset.py
```

**Outputs:**
- `./data/processed_features.pkl` - Processed MFCC features
- Console: Dataset statistics and processing progress

---

#### 2. `vae.py`
**Purpose:** VAE model architectures  
**What it does:**
- Defines Basic VAE class (fully connected)
- Defines ConvVAE class (convolutional for spectrograms)
- Implements VAE loss function (reconstruction + KL divergence)
- Provides reparameterization trick

**Note:** This is a module file, not executed directly. Used by training scripts.

---

#### 3. `train.py`
**Purpose:** Train basic VAE model  
**What it does:**
- Loads processed audio features
- Trains Basic VAE (40D → 512 → 256 → 32D latent)
- Saves trained model and latent features
- Generates training history plots

**Execution:**
```bash
python train.py
```

**Outputs:**
- `./models/vae_model.pt` - Trained VAE model
- `./data/latent_features.npy` - Latent representations (N×32)
- `./data/labels.npy` - Genre labels
- `./results/training_history.png` - Training curves
  * Total loss over epochs
  * Reconstruction loss over epochs
  * KL divergence over epochs

---

#### 4. `clustering.py`
**Purpose:** Clustering and evaluation  
**What it does:**
- K-Means clustering on VAE latent features
- PCA + K-Means baseline comparison
- Computes evaluation metrics:
  * Silhouette Score
  * Calinski-Harabasz Index
  * Davies-Bouldin Index (optional)
  * Adjusted Rand Index (if labels available)
  * Normalized Mutual Information
  * Cluster Purity

**Execution:**
```bash
python clustering.py
```

**Outputs:**
- `./results/clustering_metrics.csv` - All metrics in table format
  * Columns: method, silhouette, calinski_harabasz, davies_bouldin, ARI, NMI, purity
  * Rows: VAE+K-Means, PCA+K-Means, etc.
- Console: Formatted comparison table

---

#### 5. `visualize.py`
**Purpose:** Create visualizations for Easy task  
**What it does:**
- t-SNE visualization of latent space (colored by genre)
- t-SNE visualization colored by predicted clusters
- UMAP visualization
- Cluster distribution heatmaps
- Metrics comparison bar charts

**Execution:**
```bash
python visualize.py
```

**Outputs:**
- `./results/tsne_vae_true_labels.png` - t-SNE colored by true genres
- `./results/tsne_vae_clusters.png` - t-SNE colored by K-Means clusters
- `./results/tsne_pca_true_labels.png` - t-SNE of PCA features
- `./results/umap_vae_true_labels.png` - UMAP visualization
- `./results/cluster_distribution_vae.png` - Confusion matrix heatmap
- `./results/cluster_distribution_pca.png` - PCA confusion matrix

---

#### 6. `main.ipynb`
**Purpose:** Jupyter notebook with complete Easy task pipeline  
**What it does:**
- Runs all Easy task steps in sequential cells
- Includes explanations and intermediate outputs
- Useful for step-by-step understanding

**Execution:**
```bash
jupyter notebook main.ipynb
```

**Outputs:** Same as running individual scripts above

---

## Medium Task Files

### Text Feature Extraction

#### 7. `lyrics_fetcher.py`
**Purpose:** Download song lyrics from Genius API  
**What it does:**
- Connects to Genius API (requires API key)
- Fetches lyrics for FMA tracks by title and artist
- Saves lyrics data with metadata
- Handles rate limiting and retries

**Execution:**
```bash
python lyrics_fetcher.py
```

**Prerequisites:** 
- Get free API key from https://genius.com/api-clients
- Paste when prompted

**Outputs:**
- `./data/lyrics.json` - Lyrics data for ~500 tracks
  * Format: {track_id: {lyrics, title, artist, success, url}}
- Console: Progress and success rate

---

#### 8. `text_features.py`
**Purpose:** Convert lyrics/metadata to text embeddings  
**What it does:**
- Uses Sentence-Transformers (384D embeddings) OR TF-IDF
- Extracts features from lyrics (if available)
- Extracts features from metadata (title + artist + tags) as fallback
- Creates hybrid text representation

**Execution:**
```bash
python text_features.py
```

**Interactive:** Choose method (1=Sentence-Transformer, 2=TF-IDF)

**Outputs:**
- `./data/text_features.pkl` - Text embeddings
  * Contains: features (N×384), track_ids, metadata
  * Metadata includes source info (lyrics vs metadata)
- Console: Statistics on lyrics vs metadata coverage

---

#### 9. `hybrid_features.py`
**Purpose:** Combine audio and text features  
**What it does:**
- Matches audio and text features by track_id
- Creates multiple feature variants:
  * Concatenated (audio + text)
  * Separate (for multi-encoder VAE)
  * Audio-only
  * Text-only
- Handles missing data gracefully

**Execution:**
```bash
python hybrid_features.py
```

**Outputs:**
- `./data/features_audio_only.pkl` - Audio features only
- `./data/features_text_only.pkl` - Text features only
- `./data/features_concatenated.pkl` - Combined (N×424: 40+384)
- `./data/features_separate.pkl` - Separate audio and text arrays
- Console: Match statistics and feature dimensions

---

### Advanced Model Training

#### 10. `train_conv_vae.py`
**Purpose:** Train Convolutional VAE on spectrograms  
**What it does:**
- Extracts 2D MFCC spectrograms (20×128)
- Trains ConvVAE with convolutional encoder/decoder
- Better captures temporal structure in audio
- Saves model and latent features

**Execution:**
```bash
python train_conv_vae.py
```

**Outputs:**
- `./data/mfcc_spectrograms.pkl` - 2D MFCC data
- `./models/conv_vae_model.pt` - Trained ConvVAE
- `./data/conv_latent_features.npy` - ConvVAE latent features (N×32)
- `./data/conv_labels.npy` - Corresponding labels
- `./results/conv_vae_training_history.png` - Training curves

---

#### 11. `train_multimodal_vae.py`
**Purpose:** Train Multimodal VAE with audio + text  
**What it does:**
- Uses separate encoders for audio and text
- Fuses representations in latent space
- Trains on hybrid features
- Learns joint audio-text representation

**Execution:**
```bash
python train_multimodal_vae.py
```

**Outputs:**
- `./models/multimodal_vae_model.pt` - Trained multimodal VAE
- `./data/multimodal_latent_features.npy` - Hybrid latent features (N×32)
- `./data/multimodal_labels.npy` - Labels
- `./results/multimodal_training_history.png` - Training curves (not in base code but can be added)

---

### Advanced Evaluation

#### 12. `clustering_advanced.py`
**Purpose:** Comprehensive clustering comparison  
**What it does:**
- Loads all feature types (Basic VAE, ConvVAE, Multimodal, PCA, Raw)
- Runs multiple clustering algorithms:
  * K-Means
  * Agglomerative Clustering
  * DBSCAN
- Computes all 6 metrics for each combination
- Analyzes best methods and comparisons

**Execution:**
```bash
python clustering_advanced.py
```

**Outputs:**
- `./results/clustering_metrics_all.csv` - Complete results table
  * Rows: All method×algorithm combinations (~20-30 rows)
  * Columns: method, silhouette, CH, DB, ARI, NMI, purity
- Console: Detailed analysis including:
  * Top 5 methods
  * Best per feature type
  * VAE architecture comparison
  * Multimodal benefit analysis

---

#### 13. `visualize_advanced.py`
**Purpose:** Advanced visualizations for Medium task  
**What it does:**
- Side-by-side t-SNE comparison of all methods
- Metrics heatmap (all methods × all metrics)
- Architecture comparison charts
- Clustering algorithm comparison
- Comprehensive summary figure

**Execution:**
```bash
python visualize_advanced.py
```

**Outputs:**
- `./results/tsne_comparison.png` - Grid of t-SNE plots for all methods
- `./results/metrics_heatmap.png` - Heatmap showing all metrics
  * Raw values + Normalized values side-by-side
- `./results/architecture_comparison.png` - VAE architectures compared
- `./results/clustering_algorithms_comparison.png` - Algorithms compared
- `./results/summary_figure.png` - Comprehensive 3×3 summary
  * Top methods, t-SNE plots, metrics, statistics

---

## Hard Task Files

### Beta-VAE Implementation

#### 14. `beta_vae.py`
**Purpose:** Train Beta-VAE for disentangled representations  
**What it does:**
- Implements Beta-VAE (VAE with weighted KL divergence)
- Trains with multiple beta values: β ∈ {0.5, 1.0, 4.0, 10.0}
- Analyzes disentanglement quality
- Compares reconstruction-disentanglement trade-off

**Key Concept:**
- Loss = Reconstruction + β × KL_Divergence
- β > 1: More disentangled (better clustering)
- β < 1: Better reconstruction (less disentangled)
- β = 1: Standard VAE

**Execution:**
```bash
python beta_vae.py
```

**Outputs:**

**Models:**
- `./models/beta_vae_beta_0.5.pt` - Beta-VAE with β=0.5
- `./models/beta_vae_beta_1.0.pt` - Standard VAE (β=1.0)
- `./models/beta_vae_beta_4.0.pt` - Disentangled VAE (β=4.0)
- `./models/beta_vae_beta_10.0.pt` - Highly disentangled (β=10.0)

**Features:**
- `./data/beta_vae_latent_beta_0.5.npy` - Latent features (β=0.5)
- `./data/beta_vae_latent_beta_1.0.npy` - Latent features (β=1.0)
- `./data/beta_vae_latent_beta_4.0.npy` - Latent features (β=4.0)
- `./data/beta_vae_latent_beta_10.0.npy` - Latent features (β=10.0)
- `./data/beta_vae_labels.npy` - Corresponding labels

**Visualizations:**
- `./results/beta_vae_comparison.png` - Training dynamics comparison
  * Total loss vs epoch for all β values
  * Reconstruction loss vs epoch
  * KL divergence vs epoch
  * Final loss components bar chart

**Console Output:**
- Final losses for each β
- Disentanglement metrics (correlation, variance, active dimensions)

---

#### 15. `clustering_hard.py`
**Purpose:** Comprehensive Hard task evaluation  
**What it does:**
- Loads ALL feature variants including all Beta-VAEs
- Runs K-Means, Agglomerative, DBSCAN on all
- Computes all 6 metrics for every combination
- Detailed analysis:
  * Best overall method
  * Best beta value
  * Disentanglement benefit
  * Multimodal benefit
  * Architecture comparison
- Creates LaTeX summary table for report

**Execution:**
```bash
python clustering_hard.py
```

**Outputs:**
- `./results/clustering_metrics_hard_task.csv` - Complete results
  * All methods including 4 Beta-VAEs × 3 algorithms
  * ~40-50 rows total
  * All 6 metrics computed
- `./results/hard_task_summary_table.tex` - LaTeX table
  * Pre-formatted for direct inclusion in report
  * Includes key methods with all metrics

**Console Output:**
- Top 10 methods overall
- Beta-VAE analysis table
- VAE architecture comparison
- Clustering algorithm comparison
- Multimodal benefit analysis
- Disentanglement analysis
- Key findings summary

---

#### 16. `visualize_hard.py`
**Purpose:** Visualizations for Hard task  
**What it does:**
- Beta-VAE latent space comparison
- Disentanglement quality analysis
- Hard task performance summary
- Trade-off visualizations

**Execution:**
```bash
python visualize_hard.py
```

**Outputs:**

**Visualizations:**
- `./results/beta_vae_latent_comparison.png` - 2×2 grid
  * t-SNE plot for β=0.5
  * t-SNE plot for β=1.0
  * t-SNE plot for β=4.0
  * t-SNE plot for β=10.0
  * All colored by true genres

- `./results/disentanglement_analysis.png` - 2×2 analysis
  * Average correlation vs β (lower = more disentangled)
  * Variance statistics vs β
  * Active dimensions bar chart
  * Disentanglement-reconstruction trade-off scatter

- `./results/hard_task_performance_summary.png` - Comprehensive 3×3 figure
  * Top 10 methods bar chart
  * Beta-VAE performance curve
  * Architecture comparison
  * Clustering algorithm comparison
  * All metrics for best method
  * Summary statistics box

---

## Helper Scripts

#### 17. `test_quick.py`
**Purpose:** System verification before running pipeline  
**What it does:**
- Tests all imports
- Verifies PyTorch and CUDA
- Tests VAE model forward pass
- Tests training pipeline
- Tests clustering
- Tests visualization
- Checks data directory structure

**Execution:**
```bash
python test_quick.py
```

**Outputs:** Console messages indicating pass/fail for each test

---

#### 18. `run_all.py`
**Purpose:** Master execution script for Easy + Medium tasks  
**What it does:**
- Provides interactive menu
- Can run Easy task only OR Easy + Medium
- Executes scripts in correct order
- Shows progress and timing
- Displays summary at end

**Execution:**
```bash
python run_all.py
```

---

#### 19. `run_hard_task.py`
**Purpose:** Master execution script for Hard task  
**What it does:**
- Runs Beta-VAE training
- Runs comprehensive evaluation
- Runs Hard task visualizations
- Shows progress and summary
- Displays marks breakdown

**Execution:**
```bash
python run_hard_task.py
```

---

#### 20. `requirements.txt`
**Purpose:** Python dependencies list  
**Contents:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
torch==2.0.1
librosa==0.10.0
matplotlib==3.7.2
seaborn==0.12.2
umap-learn==0.5.3
sentence-transformers
lyricsgenius
```

---