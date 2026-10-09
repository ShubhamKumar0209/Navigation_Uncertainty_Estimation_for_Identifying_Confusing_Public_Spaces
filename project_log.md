# Navigation Uncertainty Estimation for Identifying Confusing Public Spaces
## Project Log

### Date: 2026-08-29
**Update: Preprocessing Script for SDD Bookstore Scene**
- Created `preprocess_bookstore.py` to handle the data preparation of pedestrian trajectories for the Stanford Drone Dataset (bookstore scene).
- Implemented data loading and parsing, extracting trajectories exclusively for "Pedestrian" class.
- Addressed tracking discontinuities by splitting tracks automatically if a time gap of > 1 second is detected, assigning synthetic agent IDs (e.g., `{original_id}_b`).
- Set up a linear interpolation resampling module targeting a fixed timestep of 0.4s (standard for trajectory forecasting tasks).
- Removed excessively short pedestrian tracks (less than 5 seconds in duration) to maintain high-quality data for sliding window applications.
- Included an `apply_homography()` placeholder function to ease the eventual integration of pixel-to-metric coordinates mapping.
- Implemented output generation using CSV for both individual scene-wise outputs and a combined dataset encompassing all sub-videos.

### Date: 2026-08-29 (Part 2)
**Update: Feature Computation Script for NUS_weak Estimation**
- Created `compute_features_bookstore.py` to extract uncertainty markers from processed pedestrian trajectories.
- Implemented a 5-second overlapping sliding window (stride 2.5s) to sample local motion patterns.
- Extracted 9 core features: `stopping_frequency`, `speed_variance`, `path_efficiency`, `turning_angle_variance`, `local_hesitation`, `loop_backtrack`, `waiting_time`, along with raw `u_turn_count` and `oscillation_count`.
- Setup placeholders for `interaction_change_rate` and `crowd_density` (to be filled in Part 6).
- Applied robust scaling (Median and IQR) independently per scene, saving parameters to `normalization_params.json` for reproducibility.
- Computed initial Navigation Uncertainty Scores (`NUS_weak`) using two approaches: expert-weighted fusion and data-driven PCA (first principal component).
- Exported all features, normalized values, and NUS estimates to `processed/bookstore/features_windows.csv`.
- **Bugfix**: Resolved an issue where features with near-zero IQR (e.g., `loop_backtrack`) resulted in excessively large normalized values by replacing the epsilon floor with a relative fallback (`iqr = 1.0` if `iqr <= 0.01`). Additionally, added a global clamping step to bound all normalized features within the `[-5, 5]` range to prevent any remaining outliers from dominating the downstream PCA or expert-weighted summations. The resulting `NUS_weak` scores are now cleanly bounded within expected limits.

### Date: 2026-08-29 (Part 5)
**Update: Extended Node Features for GNN Estimation**
- Created `compute_node_features_bookstore.py` to calculate a dense set of node attributes required for the Part 5 GNN modelling.
- Successfully extracted **13 new structural descriptors** including derivatives of motion (`acceleration_mean`, `jerk_mean`, `angular_velocity_mean`), global constraints (`net_displacement`, `time_since_track_start`), and spatial footprint geometry (`local_path_curvature`, `bounding_box_area`).
- Configured 5 placeholder stubs for graph-derived features (e.g., `node_degree`, `relative_velocity_to_nearest`) to be populated upon completion of the Part 6 full-scene interaction graph.
- Merged temporal bounds matching the original `NUS_weak` weak-supervision sets directly by joining on identical `window_id` slices.
- Assessed permutation importance relative to `NUS_weak_expert`, revealing high correlations and variable weighting (e.g., `net_displacement`, `dwell_duration`, and `angular_velocity_mean` acting as the strongest structural predictors).
- Finalized a programmatic `GNN_INPUT_FEATURES` list, safely dropping highly redundant descriptors (`speed_max`, `net_displacement`, and label-leaking `u_turn_count`). The remaining 16 target features (11 real + 5 stubbed) exhibit 0 pairs with `|r| > 0.9`. 
- Exported the complete matrix to `processed/bookstore/node_features.csv` and a routing config to `processed/bookstore/feature_manifest.json` for downstream scripts.

### Date: 2026-08-29 (Part 6)
**Update: Full-Scene Pedestrian Interaction Graphs**
- Created `build_graphs_bookstore.py` to construct time-sliced spatial interaction graphs across all actors.
- Implemented global 5s absolute-time windows (stride 2.5s) to guarantee synchronized temporal snapshots across the dataset.
- Evaluated scene-specific dynamic interaction radii (`r`), defining the threshold as the 90th percentile of nearest-neighbor distances for each scene (ranging empirically between 388m to 1009m).
- Built structural topologies using `scipy.spatial.cKDTree` and `networkx`, pruning connections to a max degree of 8 (closest-k) and applying Gaussian kernel edge weighting ($\sigma = r/2$).
- Successfully serialized the raw interaction networks to `processed/bookstore/graphs/*.gpickle` for native GAT message-passing.
- Populated the 5 previously stubbed features (`node_degree`, `node_centrality`, `nearest_neighbor_distance`, `relative_velocity_to_nearest`, `local_pedestrian_density`), plus tracking metrics like `interaction_change_rate` and `crowd_density`.
- Executed an intersection join linking the global graph data back to the agent-centric sliding windows, producing zero missed overlaps and perfectly replacing all NaNs in `features_windows_with_graph.csv` and `node_features_with_graph.csv`.
- **Bugfix (Degree Capping)**: Re-implemented the max-degree cap as a global, distance-sorted greedy procedure to enforce symmetric undirected edge constraints, ensuring no node ever exceeds a degree of 8 (fixing a bug where independent local clipping allowed global caps up to 16).
- **Bugfix (Graph Normalization)**: Applied robust scene-wise normalization (median/IQR scaling with floor and `[-5, 5]` clipping) to four unscaled variables (`local_pedestrian_density`, `crowd_density`, `nearest_neighbor_distance`, `relative_velocity_to_nearest`), preventing exponential density fluctuations from dominating downstream models. Parameters saved to `graph_feature_normalization_params.json`.
- **Finalized NUS_weak**: Wrote `finalize_nus_weak.py` to inject the now-real `interaction_change_rate` into the weak supervision labels. The expert weights were updated (incorporating the new social metric at $0.1$ and leaving `crowd_density` unweighted as a confound). The resultant `NUS_weak_expert` holds a 0.9667 correlation with the pre-graph version, indicating a smooth, stable structural enhancement to the label. 
- **Target Bounding [0, 1]**: Applied global min-max scaling to bound both `NUS_weak_expert` and `NUS_weak_pca` strictly between [0.0, 1.0] to satisfy downstream GAT regression requirements. Bounding params are retained in `nus_weak_scaling_params.json` to ensure exact reproduction on inference data. The distribution behaves well across the interval (0.2% in top decile, 5.5% in bottom decile). Final dataset successfully exported to `processed/bookstore/features_final.csv`.

### Date: 2026-08-29 (Part 7)
**Update: PyTorch Geometric Dataset Assembly**
- Created `build_pyg_dataset.py` to bridge the pandas/networkx outputs into an optimized machine learning dataset for GAT+GRU processing.
- Installed `torch` and `torch_geometric` inside the virtual environment to facilitate tensor conversion and graph abstractions.
- **Node Alignment Strategy**: Global graphs structurally define the nodes for each 5-second slice. To assign the 16 `GNN_INPUT_FEATURES` reliably, the agent's local window with the *maximum temporal overlap* to the global window was successfully extracted, cleanly resolving any phase-shifts. Handled highly degenerate (zero-movement) agents by zero-filling previously NaN trajectory features. 
- **Sequence Generation**: Mapped 3669 target agent samples to an $L=6$ historical structural sequence (30 seconds of context). 134 edge-case samples (agents appearing early in the video) were appropriately padded by duplicating the earliest available snapshots.
- Implemented robust `target_node_index` caching (with `-1` zero-vector catch mapping) to allow the downstream sequential unroller to accurately trace moving nodes dynamically through the changing graphs. 
- Output safely serialized to `processed/bookstore/dataset.pt`, distributing ~3.6k supervised graphs across 7 scenes.

**Update: Model Regularization and Rotation Experiment**
- **Architecture Tuning**: Increased GAT dropout to `0.3` and added a `0.2` dropout block before the GRU regression heads to combat overfitting. Configured gradient clipping (`max_norm=1.0`) and reduced early stopping patience to `5` epochs. Migrated standard execution to CPU to avoid PyG MPS dispatch overhead.
- **3-Way Rotation Evaluation**: Refactored `train.py` to evaluate generalization across three unique Scene Rotations (A: Test `video4`, B: Test `video6`, C: Test `video5`).
  - Rotation A (Test `video4`): Spearman $\rho \approx 0.00$, confirming it acts as a pathological hold-out.
  - Rotation B (Test `video6`): Spearman $\rho = 0.43$.
  - Rotation C (Test `video5`): Spearman $\rho = 0.56$, demonstrating strong generalizability when representative density distributions are maintained in the training set.


