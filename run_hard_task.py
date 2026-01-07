"""
Master script to run Hard Task
Executes Beta-VAE training, evaluation, and visualization
"""

import subprocess
import time

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

def main():
    """Execute Hard Task pipeline"""
    print("="*80)
    print("HARD TASK EXECUTION")
    print("="*80)
    print("\nThis will:")
    print("  1. Train Beta-VAEs with β = [0.5, 1.0, 4.0, 10.0]")
    print("  2. Evaluate all methods comprehensively")
    print("  3. Create visualizations for report")
    print(f"\nEstimated time: ~2 hours")
    
    input("\nPress Enter to continue...")
    
    start_time = time.time()
    
    steps = [
        ("python beta_vae.py", "Train Beta-VAEs (90 min)"),
        ("python clustering_hard.py", "Comprehensive evaluation (20 min)"),
        ("python visualize_hard.py", "Create visualizations (10 min)")
    ]
    
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        if not success:
            print(f"\n✗ Hard task failed at: {desc}")
            return False
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 HARD TASK COMPLETE!")
    print("="*80)
    print(f"Total execution time: {total_time/60:.1f} minutes")
    
    print("\n📁 Generated Files:")
    print("\nModels:")
    print("  - ./models/beta_vae_beta_0.5.pt")
    print("  - ./models/beta_vae_beta_1.0.pt")
    print("  - ./models/beta_vae_beta_4.0.pt")
    print("  - ./models/beta_vae_beta_10.0.pt")
    
    print("\nData:")
    print("  - ./data/beta_vae_latent_beta_*.npy (4 files)")
    
    print("\nResults:")
    print("  - ./results/clustering_metrics_hard_task.csv")
    print("  - ./results/hard_task_summary_table.tex")
    
    print("\nVisualizations:")
    print("  - ./results/beta_vae_comparison.png")
    print("  - ./results/beta_vae_latent_comparison.png")
    print("  - ./results/disentanglement_analysis.png")
    print("  - ./results/hard_task_performance_summary.png")
    
    print("\n" + "="*80)
    print("MARKS BREAKDOWN")
    print("="*80)
    print("✅ Easy Task: 20 marks")
    print("✅ Medium Task: 25 marks")
    print("✅ Hard Task: 25 marks")
    print("✅ Evaluation Metrics: 10 marks")
    print("✅ Visualization: 10 marks")
    print("📝 Report Quality: 10 marks (write good report!)")
    print("💻 GitHub Repository: 10 marks (clean up repo!)")
    print("\n🎯 TOTAL: 100 marks possible!")
    print("="*80)
    
    print("\n✅ Next Steps:")
    print("1. Review all results in ./results/")
    print("2. Check clustering_metrics_hard_task.csv")
    print("3. Look at visualizations")
    print("4. START WRITING REPORT (most important now!)")
    print("5. Use hard_task_summary_table.tex in report")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 Ready for report writing!")
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")