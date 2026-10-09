# AGENT PROMPT — VS Code Copilot / Claude Opus 4.6
## Task: Implement & Validate the Full Patent System in `main.ipynb`

---

## ROLE & OBJECTIVE

You are an expert ML systems engineer and research scientist. Your task is to build a **complete, runnable, well-documented Jupyter notebook** (`main.ipynb`) that implements, demonstrates, and validates every novel technical component of the following patent:

**Title:** *Graph-Based Pedestrian Navigation Uncertainty Estimation for Identifying Confusing Public Spaces*

The notebook must:
1. Implement all five patented modules from scratch in Python
2. Process provided pedestrian trajectory dataset (e.g., Stanford Drone Dataset snippet) or generate synthetic but realistic pedestrian telemetry data
3. Produce a **publication-quality technical report** with charts, metrics, and results that substantiate each novelty claim
4. Be structured to satisfy the **Indian Patents Act, 1970 (as amended)** — specifically demonstrating **novelty (Section 2(1)(l))**, **inventive step (Section 2(1)(ja))**, and **industrial applicability (Section 2(1)(ac))** for software-based inventions under the **Controller General's Guidelines for Examination of Computer Related Inventions (CRI), 2017**

---

## CONTEXT — THE FIVE PATENTED MODULES

### Module 1: Kinematic Feature Extraction & Weak Label Generation
- At each timestep `t`, extract kinematic features from trajectory windows:
  - Stopping frequency (frames below speed threshold)
  - Speed variance
  - Path efficiency (displacement vs path length)
  - Turning angle variance
  - Loop/backtrack events
- Calculate a continuous **Navigation Uncertainty Score (NUS_weak)** via expert-weighting or PCA formulation to serve as weak supervision.
- **Indian Patent Novelty Hook:** Computing a continuous wayfinding uncertainty proxy directly from multi-modal kinematic irregularities (not merely binary anomaly detection) is novel over cited prior art.

### Module 2: Structural Node Feature Extraction
- Compute a dense set of node attributes for graph estimation:
  - Motion derivatives: `acceleration_mean`, `jerk_mean`, `angular_velocity_mean`
  - Global constraints: `net_displacement`, `time_since_track_start`
  - Spatial footprint: `local_path_curvature`, `bounding_box_area`
- Prune highly redundant features using correlation analysis and Permutation Importance relative to `NUS_weak`.
- **Indian Patent Novelty Hook:** Generating comprehensive node profiles that capture individual behavior context for graph input over moving windows.

### Module 3: Dynamic Spatial-Temporal Interaction Graph Construction
- Construct time-sliced spatial interaction graphs over the scene.
- Use a **dynamic interaction radius** thresholded at the 90th percentile of nearest-neighbor distances for each scene.
- Build structural topologies mapping pedestrians as nodes and spatial proximities as edges (capped k-NN + radius hybrid) with Gaussian kernel edge weighting.
- Extract tracking metrics like `interaction_change_rate` and `local_pedestrian_density`.
- **Indian Patent Novelty Hook:** Combining absolute-time global windows with dynamic interaction radii to encapsulate pedestrian social topologies in confusing spaces.

### Module 4: Spatio-Temporal Graph Neural Network (ST-GNN) Modeler
- Feed PyTorch Geometric graphs into a Graph Attention Network (GAT) coupled with a temporal aggregator (e.g., GRU or Temporal Attention) to map inputs to the `NUS` score.
- Model sequence generation tracking moving nodes dynamically through the varying graphs (e.g., 30-second structural sequence).
- Output the predicted continuous **Navigation Uncertainty Score (NUS)** ∈ [0,1].
- **Indian Patent Novelty Hook:** Fusing GAT with temporal memory (GRU/Attention) specifically regressed against continuous trajectory uncertainty instead of traditional ADE/FDE future coordinate forecasting.

### Module 5: Spatial Heatmap & Explainability Engine (EAI)
- Aggregate the pedestrian-level NUS predictions into a uniform grid over the physical space to generate a **Spatial Heatmap** of environment-level wayfinding difficulty.
- Apply SHAP and/or GNNExplainer to map feature importance on predicted high-NUS events.
- **Indian Patent Novelty Hook:** Combining aggregate continuous spatial heatmapping with per-instance feature attributions to provide actionable diagnostic feedback for urban space design.

---

## NOTEBOOK STRUCTURE — IMPLEMENT EXACTLY IN THIS ORDER

### Cell 0 — Title & Patent Header (Markdown)
```markdown
# Patent Implementation: Graph-Based Pedestrian Navigation Uncertainty Estimation
## for Identifying Confusing Public Spaces
### Module Implementation, Validation & Technical Report
**Indian Patent Application — Technical Demonstration Notebook**
Date: [auto-generated]
```

### Cell 1 — Imports & Environment Setup
Install and import all required packages. Use:
```python
# Core
numpy, pandas, scipy, scikit-learn
# ML
torch, torch_geometric, xgboost, shap
# Visualization
matplotlib, seaborn, plotly (for interactive)
# Graph
networkx
# Utilities
tqdm, warnings, logging, datetime, collections
```
Set random seeds for reproducibility: `SEED = 42`

### Cell 2 — Synthetic/Sample Trajectory Data Loader
Build a `TrajectoryGenerator` class that simulates **realistic pedestrian trajectory data** across various scenes over a configurable time horizon if real data is unavailable.
- Needs to produce scenarios for **three distinct behaviors**:
  - **Normal:** High-efficiency paths, consistent speed.
  - **Crowded Queueing:** Low speed, high density, but efficient progression (not confused).
  - **Confused Wayfinding:** High turning variance, looping, stopping, low path efficiency.
- Generate fields: `timestamp`, `agent_id`, `x_world`, `y_world`.

### Cell 3 — Module 1: Kinematic Feature Extraction & Weak Label Generation
Implement `KinematicFeatureExtractor` class:
- Extract local windows (e.g., 5 seconds with 2.5s stride).
- Compute `stopping_frequency`, `speed_variance`, `path_efficiency`, `turning_angle_variance`, `loop_backtrack`.
- Output `NUS_weak` using an expert-weighted combination.

### Cell 4 — Module 2: Structural Node Feature Extraction
Implement `NodeFeatureExtractor` class:
- Compute structural attributes (`acceleration_mean`, `jerk_mean`, `local_path_curvature`, etc.).
- Prune highly correlated features (|r| > 0.9).
- Prepare final `node_features` array.

### Cell 5 — Module 3: Dynamic Spatial-Temporal Interaction Graph Construction
Implement `InteractionGraphBuilder` class:
- Implement global 5s absolute-time windows.
- Build `networkx` graphs using `scipy.spatial.cKDTree` for nearest neighbors and capped degree radius logic.
- Extract `interaction_change_rate` and merge back to local windows.
- Scale structural variables safely into [-5, 5].

### Cell 6 — Module 4: Spatio-Temporal Graph Neural Network (ST-GNN) Modeler
Implement `NavigationUncertaintyGNN` class:
- Input layers: Node features + Adjacency matrices.
- Architecture: 2 GAT layers + GRU temporal sequence aggregator + MLP output head.
- Loss function: Huber Loss + Pairwise Ranking loss.
- Output: `predicted_NUS` score.

### Cell 7 — Module 5: Spatial Heatmap & Explainability Engine
Implement `HeatmapAndExplainability` class:
- Accumulate window-level `predicted_NUS` values into physical grid cells.
- Generate smoothed Gaussian kernel density heatmaps.
- Utilize SHAP or feature permutation to identify which kinematic inputs drove high NUS scores in flagged grid cells.

### Cell 8 — End-to-End Pipeline Integration
Implement `NavigationUncertaintyPipeline` class that:
1. Ingests the trajectory stream.
2. Calls each module's update/build steps sequentially.
3. Feeds graphs to the trained GNN.
4. Stores intermediate and final predictions for grid analysis.

### Cell 9 — Evaluation & Metrics (CRITICAL for Patent)
Compute and display the following:
- Correlation metrics (Spearman $\rho$, MAE, RMSE) of predicted NUS vs NUS_weak.
- Rank-based validation (precision at top 10% highest NUS windows).
- Cross-scene validation to demonstrate spatial generalization (e.g. Train on Scene A, Test on Scene B).

**Ablation Study (MANDATORY):**
Run variants to prove inventive step:
| System Variant | Spearman ρ | RMSE |
|---|---|---|
| No Graph Structure (LSTM only) | | |
| No Temporal Aggregation (GAT only) | | |
| PCA-derived Labeling vs Expert Weighting | | |
| Full System (GAT + Temporal) | | |

### Cell 10 — Visualizations & Report Figures (ALL REQUIRED)

**Figure 1 — System Architecture Diagram**
Programmatic block diagram of the GNN processing flow.

**Figure 2 — Kinematic Feature Distributions**
Violin plots of kinematic metrics comparing normal vs top 10% confused windows.

**Figure 3 — Weak Label (NUS_weak) Correlation**
Scatter plot of predicted NUS vs weak labels.

**Figure 4 — Spatial Interaction Graph Snapshots**
Plot the nodes (pedestrians) and edges (social connections) over a top-down view for a specific crowded time slice.

**Figure 5 — Spatial Navigation Uncertainty Heatmap**
2D Top-down heatmap of aggregated NUS scores (red = high uncertainty).

**Figure 6 — Explanation Feature Importance (SHAP)**
Bar chart showing top contributing features to the model's predictions.

### Cell 11 — Patent Technical Report (Markdown + Code Output)

Generate a structured technical report mapping results to Indian patent requirements:

#### Section A: Technical Field & Industrial Applicability
Applicable to architectural design, transit hubs, smart cities.

#### Section B: Novelty Analysis (Indian Patents Act, Section 2(1)(l))
Reference the following prior arts and papers and establish novelty over them (the model focuses on continuous individual navigation uncertainty using multi-modal kinematics instead of crowd flow anomaly or basic route prediction):
- US11495055, US20200324794A1, US20220292867A1, US20230257003A1
- US 10971003 B2, US 2025/0044866 A1, US 7496445 B2, US 7539576 B2
- CN110827316A, CN117315724A
- Scientific Literature: Uncertainty Estimation of Pedestrian Future Trajectory Using Bayesian Approximation, machines-10-00703-v2, s43762-026-00258-5, ijgi-10-00146, 1-s2.0-S187705092100630X-main, 202302L004.

#### Section C: Inventive Step (Indian Patents Act, Section 2(1)(ja))
Present the ablation study showing that integrating explicit structural context (GNN) and temporal dynamics produces a non-obvious synergistic effect on predicting continuous navigation uncertainty.

#### Section D: Technical Advancement Over Prior Art
Comparison table: Future trajectory forecasting (ADE/FDE metrics) vs Anomaly Detection vs Present Invention (NUS Score Heatmaps).

#### Section E: Industrial Applicability (CRI Guidelines 2017)
Produces a technical effect: enabling automated auditing of physical infrastructure layouts and quantitative wayfinding evaluation.

#### Section F: Claims Support Matrix
Table mapping method claims to implementation steps and figures.

### Cell 12 — Save All Outputs

```python
# Save results
results_df.to_csv('results/pipeline_results.csv', index=False)
# Save all figures
for fig_name, fig in figures.items():
    fig.savefig(f'results/figures/{fig_name}.png', dpi=300, bbox_inches='tight')
# Save trained models
import torch
torch.save(gnn_model.state_dict(), 'models/best_model_gnn.pt')

print("All outputs saved.")
```

---

## CODE QUALITY REQUIREMENTS
1. Detailed docstrings explaining role in the patent claim.
2. Inline comments for mathematical operations.
3. No hardcoded magic numbers (use a `CONFIG` dict).
4. `SEED = 42` for reproducibility.
5. Print statements summarizing outputs.

---

## EXPECTED PERFORMANCE TARGETS
- Spearman Rank Correlation $\rho$ ≥ 0.50 on unseen scenes.
- Heatmap visually aligns with actual structural chokepoints in data.
- Ablation models confirm the superiority of the Full ST-GNN architecture.

---

## FILE STRUCTURE TO CREATE
```
project/
├── main.ipynb                    ← THE MAIN NOTEBOOK
├── results/
│   ├── pipeline_results.csv
│   └── figures/
│       ├── fig1_architecture.png
│       ├── fig2_kinematics.png
│       ├── fig3_nus_correlation.png
│       ├── fig4_graphs.png
│       ├── fig5_heatmap.png
│       └── fig6_shap.png
├── models/
│   └── best_model_gnn.pt
└── README.md                     ← auto-generated summary
```

---

## FINAL INSTRUCTION TO AGENT
Begin by creating `main.ipynb` with Cell 0 (title) and Cell 1 (imports). Then implement each cell in sequence. After each module cell (Cells 3–7), immediately test it on simulated or real data.
Do not summarize or skip any section. The notebook must complete without errors and produce all figures plus the full patent report in Cell 11.
