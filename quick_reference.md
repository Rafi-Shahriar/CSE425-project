# QUICK REFERENCE CARD 🚀

## Installation
```bash
pip install -r requirements.txt
```

## Fastest Way to Complete Project

### Option 1: Run Everything Automatically
```bash
python run_all.py
# Choose option 2 for Easy + Medium
# Sit back and wait ~8 hours
```

### Option 2: Run Step by Step

#### EASY TASK (2-3 hours) → 20 marks
```bash
python test_quick.py      # 2 min - verify setup
python dataset.py         # 30 min - extract audio features
python train.py           # 60 min - train VAE
python clustering.py      # 10 min - cluster & evaluate
python visualize.py       # 10 min - create plots
```

#### MEDIUM TASK (5-6 hours) → 25 marks
```bash
# Get Genius API key first: https://genius.com/api-clients

python lyrics_fetcher.py  # 60 min - download lyrics
python text_features.py   # 10 min - lyrics → embeddings
python hybrid_features.py # 10 min - combine audio + text
python train_conv_vae.py  # 60 min - train ConvVAE
python train_multimodal_vae.py  # 90 min - train hybrid VAE
python clustering_advanced.py   # 20 min - compare all methods
python visualize_advanced.py    # 20 min - create plots
```

---

## Critical Files Locations

### Your Code (15 files):
```
dataset.py, vae.py, train.py, clustering.py, visualize.py
lyrics_fetcher.py, text_features.py, hybrid_features.py
train_conv_vae.py, train_multimodal_vae.py
clustering_advanced.py, visualize_advanced.py
test_quick.py, run_all.py, main.ipynb
```

### Outputs for Report:
```
results/clustering_metrics.csv        ← Easy task metrics
results/clustering_metrics_all.csv    ← Medium task metrics
results/summary_figure.png            ← Best plot for report
results/tsne_comparison.png           ← Show all methods
results/metrics_heatmap.png           ← Compare metrics
```

---

## Common Issues & Fixes

### Out of Memory
```python
# In train.py, reduce batch size:
batch_size = 16  # instead of 32
```

### Training Too Slow
```python
# Reduce epochs:
epochs = 30  # instead of 50

# Or reduce samples:
max_samples = 300  # instead of 600
```

### No Lyrics Found
```
This is normal! Only ~500/3000 songs have lyrics.
The code uses metadata as fallback - this is fine!
```

### Genius API Errors
```
1. Check API key is correct
2. Wait 1 second between requests (rate limit)
3. Run lyrics_fetcher.py again - it resumes
```

---

## Report Writing Speed Tips

### Use This Structure (Copy-Paste Ready):

**Abstract** (5 min):
```
We implement a VAE-based clustering pipeline for hybrid music data.
We compare basic VAE, ConvVAE, and multimodal VAE on audio+text features.
Best result: [METHOD] achieves [SILHOUETTE] score.
```

**Method** (30 min):
```
1. Feature extraction: MFCC (40D) + lyrics embeddings (384D)
2. Models: Basic VAE, ConvVAE, Multimodal VAE
3. Clustering: K-Means, Agglomerative, DBSCAN
4. Metrics: Silhouette, CH, DB, ARI, NMI, Purity
```

**Results** (20 min):
```
Copy clustering_metrics_all.csv → Format as LaTeX table
Include 3 plots: summary_figure, tsne_comparison, metrics_heatmap
```

**Discussion** (15 min):
```
Multimodal VAE > Basic VAE because [fill from results]
ConvVAE captures temporal patterns better than basic
Limitation: Only ~500 songs have lyrics
```

---

## Grade Maximization Checklist

### Easy Task (20 marks):
- [x] VAE implemented
- [x] Audio features extracted
- [x] K-Means clustering
- [x] PCA baseline comparison
- [x] t-SNE visualization
- [x] Silhouette + CH metrics

### Medium Task (25 marks):
- [x] ConvVAE with spectrograms
- [x] Text features (lyrics/metadata)
- [x] Hybrid audio+text features
- [x] Multiple clustering methods
- [x] All 6 metrics computed
- [x] Comprehensive analysis

### Other (20 marks):
- [x] All metrics correct
- [x] 10+ visualizations

### Report (10 marks):
- [ ] NeurIPS format
- [ ] Clear writing
- [ ] All sections complete
- [ ] Plots included
- [ ] References cited

### Code (10 marks):
- [x] Clean structure
- [x] Comments added
- [x] README.md
- [x] requirements.txt
- [x] Reproducible

---

## Time Budget for Today

```
Hour 0-1:   Setup + Easy task start
Hour 1-2:   VAE training (Easy)
Hour 2-3:   Finish Easy task
Hour 3-4:   Get lyrics (Medium)
Hour 4-5:   Text features + hybrid
Hour 5-7:   Train ConvVAE + Multimodal VAE
Hour 7-8:   Advanced clustering + viz
Hour 8-11:  Report writing
Hour 11-12: GitHub cleanup + submission

Total: 12 hours (1 full day)
```

---

## Emergency Shortcuts (If Running Out of Time)

### Priority 1: Complete Easy Task
```bash
python dataset.py && python train.py && python clustering.py
# This gets you 20 marks minimum
```

### Priority 2: Add One Medium Feature
```bash
python train_conv_vae.py
# ConvVAE alone → +15 marks
```

### Priority 3: Write Report
```
Even with just Easy task + basic report = 60+ marks
```

---

## Key Numbers to Remember

- **Features**: 40D MFCC → 32D latent
- **Dataset**: 3000 tracks, 5 genres
- **Lyrics**: ~500 with lyrics, 2500 with metadata
- **Training**: 50 epochs ≈ 60 min each VAE
- **Expected Silhouette**: 0.3-0.5 (good)
- **Target Grade**: 80+ marks

---

## Before Submission

### Test Everything:
```bash
python test_quick.py  # Should pass all tests
```

### Check Files Exist:
```bash
ls -la data/         # Should have .npy, .pkl, .json
ls -la models/       # Should have .pt files
ls -la results/      # Should have .csv and .png files
```

### Clean Repository:
```bash
# Remove temporary files:
rm -rf __pycache__/
rm .genius_api_key  # Don't commit API key!

# Commit everything:
git add .
git commit -m "Complete VAE music clustering project"
git push
```

---

## Contact Info for Help

- **Project Document**: CSE425_04_05_ProjectDetails.pdf
- **Template**: https://www.overleaf.com/latex/templates/neurips-2024/
- **Genius API**: https://genius.com/api-clients

---

## Final Confidence Check

✅ I have FMA dataset downloaded
✅ I can run Python scripts
✅ I understand the execution order
✅ I know how to get Genius API key
✅ I have time today (10-12 hours)

**GO! START NOW! 🚀**

```bash
python test_quick.py
```