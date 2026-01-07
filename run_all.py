"""
Master script to run entire pipeline
Can run Easy task only or Easy + Medium tasks
"""

import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with error: {e}")
        return False

def run_easy_task():
    """Run Easy Task pipeline"""
    print("\n" + "="*80)
    print("STARTING EASY TASK PIPELINE")
    print("="*80)
    
    steps = [
        ("python test_quick.py", "System verification"),
        ("python dataset.py", "Audio feature extraction (30 min)"),
        ("python train.py", "Train basic VAE (60 min)"),
        ("python clustering.py", "Clustering and evaluation (10 min)"),
        ("python visualize.py", "Create visualizations (10 min)")
    ]
    
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        if not success:
            print(f"\n✗ Easy task failed at: {desc}")
            return False
    
    print("\n" + "="*80)
    print("✓ EASY TASK COMPLETE - 20 MARKS SECURED!")
    print("="*80)
    return True

def run_medium_task():
    """Run Medium Task pipeline"""
    print("\n" + "="*80)
    print("STARTING MEDIUM TASK PIPELINE")
    print("="*80)
    
    # Check if Genius API key is set up
    if not Path('.genius_api_key').exists():
        print("\n⚠ WARNING: Genius API key not found!")
        print("You'll need to enter it when lyrics_fetcher.py runs")
        print("Get your key from: https://genius.com/api-clients")
        input("\nPress Enter when ready to continue...")
    
    steps = [
        ("python lyrics_fetcher.py", "Fetch lyrics from Genius API (60 min)"),
        ("python text_features.py", "Extract text features (10 min)"),
        ("python hybrid_features.py", "Create hybrid features (10 min)"),
        ("python train_conv_vae.py", "Train ConvVAE (60 min)"),
        ("python train_multimodal_vae.py", "Train Multimodal VAE (90 min)"),
        ("python clustering_advanced.py", "Advanced clustering (20 min)"),
        ("python visualize_advanced.py", "Advanced visualizations (20 min)")
    ]
    
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        if not success:
            print(f"\n✗ Medium task failed at: {desc}")
            return False
    
    print("\n" + "="*80)
    print("✓ MEDIUM TASK COMPLETE - 25 MARKS SECURED!")
    print("="*80)
    return True

def show_summary():
    """Show summary of generated files"""
    print("\n" + "="*80)
    print("EXECUTION COMPLETE - SUMMARY")
    print("="*80)
    
    print("\n📁 Generated Files:")
    
    # Check data files
    print("\nData Files:")
    data_files = [
        './data/processed_features.pkl',
        './data/latent_features.npy',
        './data/lyrics.json',
        './data/text_features.pkl',
        './data/conv_latent_features.npy',
        './data/multimodal_latent_features.npy'
    ]
    for f in data_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (not found)")
    
    # Check models
    print("\nModels:")
    model_files = [
        './models/vae_model.pt',
        './models/conv_vae_model.pt',
        './models/multimodal_vae_model.pt'
    ]
    for f in model_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (not found)")
    
    # Check results
    print("\nResults:")
    result_files = [
        './results/clustering_metrics.csv',
        './results/clustering_metrics_all.csv',
        './results/training_history.png',
        './results/summary_figure.png'
    ]
    for f in result_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (not found)")
    
    # Count visualizations
    results_dir = Path('./results')
    if results_dir.exists():
        png_files = list(results_dir.glob('*.png'))
        print(f"\n  Total visualizations: {len(png_files)} PNG files")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review results in ./results/")
    print("2. Check metrics in CSV files")
    print("3. Write your NeurIPS report")
    print("4. Use visualizations in report")
    print("5. Clean up GitHub repo")
    print("6. Submit!")
    
    print("\n📊 Expected Grade:")
    print("  - Easy Task: 20 marks ✓")
    print("  - Medium Task: 25 marks ✓")
    print("  - Metrics: 10 marks ✓")
    print("  - Visualization: 10 marks ✓")
    print("  - Report: ~10 marks (write good report!)")
    print("  - Code: ~10 marks (clean repo!)")
    print("  - TOTAL: ~85 marks 🎯")

def main():
    """Main execution"""
    print("="*80)
    print("VAE MUSIC CLUSTERING - MASTER EXECUTION SCRIPT")
    print("="*80)
    
    print("\nWhat do you want to run?")
    print("1. Easy Task only (~2 hours)")
    print("2. Easy + Medium Tasks (~8 hours)")
    print("3. Medium Task only (assumes Easy is done)")
    
    choice = input("\nEnter choice (1/2/3) [2]: ").strip() or "2"
    
    start_time = time.time()
    
    if choice == "1":
        success = run_easy_task()
    elif choice == "2":
        easy_success = run_easy_task()
        if easy_success:
            medium_success = run_medium_task()
            success = easy_success and medium_success
        else:
            success = False
    elif choice == "3":
        success = run_medium_task()
    else:
        print("Invalid choice")
        return
    
    total_time = time.time() - start_time
    
    if success:
        print("\n" + "="*80)
        print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total execution time: {total_time/3600:.1f} hours")
        show_summary()
    else:
        print("\n" + "="*80)
        print("⚠ EXECUTION INCOMPLETE")
        print("="*80)
        print("Some steps failed. Check error messages above.")
        print("You can run individual scripts manually to debug.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        print("You can resume by running individual scripts")