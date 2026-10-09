import os
import sys
import subprocess

locations = ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        exit(1)

def main():
    print("=== Phase 1: Location-Specific Processing ===")
    for loc in locations:
        print(f"\n--- Processing {loc} ---")
        run_cmd(f"{sys.executable} preprocess.py --location {loc}")
        run_cmd(f"{sys.executable} compute_features.py --location {loc}")
        run_cmd(f"{sys.executable} compute_node_features.py --location {loc}")
        run_cmd(f"{sys.executable} build_graphs.py --location {loc}")
        
    print("\n=== Phase 2: Global Orchestration & Training ===")
    run_cmd(f"{sys.executable} finalize_nus_weak.py")
    run_cmd(f"{sys.executable} build_pyg_dataset.py")
    run_cmd(f"{sys.executable} train.py")
    
    print("\n=== Phase 3: Explainability and Heatmaps ===")
    run_cmd(f"{sys.executable} generate_heatmap.py")
    run_cmd(f"{sys.executable} explainability.py")
    
    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
