# End-to-End Pipeline Implementation Guide
This document outlines the complete data processing, feature engineering, and graph construction pipeline implemented for the `bookstore` dataset. It is designed to serve as a comprehensive template to implement the exact same pipeline for other data locations in the Stanford Drone Dataset (or similar datasets).

## 1. Preprocessing (`preprocess_<location>.py`)
The goal of this stage is to convert raw annotations into a cleaned, unified schema representing continuous pedestrian trajectories.

**Input:** Raw `annotations.txt` files for each video in the location.
**Output:** Cleaned CSV per video, and a combined `<location>_all.csv` in `processed/<location>/`.

### Steps:
1. **Load and Parse Annotations:** Read the space-separated CSV files. Ensure column names: `track_id`, `xmin`, `ymin`, `xmax`, `ymax`, `frame`, `lost`, `occluded`, `generated`, `label`.
2. **Filter Pedestrians:** Keep only rows where `label == "Pedestrian"`.
3. **Drop Lost:** Remove rows where the `lost` flag is true (these are out-of-frame or invalid coordinates).
4. **Compute Centroid and Timestamp:** 
   - `x_pixel = (xmin + xmax) / 2`
   - `y_pixel = (ymin + ymax) / 2`
   - `timestamp = frame / fps` (assume 30 FPS for SDD).
5. **Apply Homography:** Convert pixel coordinates to world metric coordinates. *(Currently implemented as a 1:1 mapping placeholder, but should be replaced with actual homography matrices).*
6. **Build Unified Schema:** Format the dataframe to: `[dataset, scene_id, agent_id, frame, timestamp, x_world, y_world, agent_type]`.
7. **Split Tracks by Gap:** For any consecutive observations of the same `agent_id` separated by a time gap `> 1.0 seconds`, split the track into separate agents (e.g., `agent_id_b`, `agent_id_c`).
8. **Resample and Filter Tracks:** 
   - Resample tracks to a fixed timestep `dt = 0.4` seconds (2.5 Hz) using linear interpolation for `x_world` and `y_world`.
   - Discard any track with a total duration shorter than `min_duration = 5.0` seconds.

## 2. Kinematic Feature Extraction (`compute_features_<location>.py`)
This stage extracts kinematic and geometric features used to estimate the Navigation Uncertainty Score (NUS_weak).

**Input:** `<location>_all.csv`
**Output:** `features_windows.csv` and `normalization_params.json`

### Steps:
1. **Window Extraction:** Group data by `scene_id` and `agent_id`. Segment each track into overlapping temporal windows.
   - `window_size = 5.0` seconds
   - `stride = 2.5` seconds
   - Minimum required samples per window = `8`.
2. **Feature Computation (per window):** Calculate velocities from `dx, dy, dt`. Compute:
   - `stopping_frequency`: Proportion of frames where speed is below `V_MIN` (e.g., 5.0 units/sec).
   - `speed_variance`: Variance of speeds scaled by mean speed squared.
   - `path_efficiency`: Ratio of straight-line distance to total path distance.
   - `turning_angle_variance`: Variance of frame-to-frame heading changes.
   - `local_hesitation`: Number of local minima in speed divided by duration.
   - `loop_backtrack`: Rate of U-turns (heading changes > 150 degrees) divided by duration.
   - `waiting_time`: Maximum contiguous run of slow speeds multiplied by `dt`, divided by duration.
   - Counts for `u_turn_count` and `oscillation_count`.
3. **Robust Normalization:** 
   - Group by `scene_id` and compute the median and IQR (Interquartile Range) for each feature.
   - Normalize: `(value - median) / IQR`. (Fallback to IQR=1.0 if IQR < 0.01).
   - Clip scaled values to `[-5, 5]`.
   - Save these parameters to a JSON file to apply during inference.

## 3. Node Feature Extraction (`compute_node_features_<location>.py`)
This stage extracts the specific features that will be fed directly into the Graph Neural Network (GNN).

**Input:** `<location>_all.csv` and `features_windows.csv` (to merge overlaps).
**Output:** `node_features.csv` and `feature_manifest.json`

### Steps:
1. **Window Extraction:** Replicate the 5.0s window / 2.5s stride segmentation.
2. **Compute GNN Features:**
   - `speed_mean`, `speed_max`
   - `acceleration_mean`, `acceleration_std` (from first derivative of speed).
   - `jerk_mean` (from second derivative of speed).
   - `heading_mean` (circular mean of heading angles), `angular_velocity_mean`.
   - `deceleration_event_count` (count of accelerations below a configurable `DECEL_THRESHOLD`, e.g., -0.5).
   - `local_path_curvature`: Average absolute curvature.
   - `bounding_box_area`: Area covered by trajectory in the window.
   - `net_displacement`, `time_since_track_start`.
3. **Correlation Pruning:** Remove highly redundant features (e.g. drop `speed_max` if |r| > 0.9 with `speed_mean`).
4. **Permutation Importance:** Fit a Random Forest Regressor against `NUS_weak_expert` (from Part 2) to rank the predictive power of node features.

## 4. Graph Construction and Graph Features (`build_graphs_<location>.py`)
This stage establishes the spatial-temporal interaction graphs between pedestrians and extracts interaction features.

**Input:** `<location>_all.csv`
**Output:** Pickle files of NetworkX graphs in `graphs/`, `graph_derived_features.csv`, and updated window features.

### Steps:
1. **Global Windows:** Slice the entire scene into global 5.0s windows (stride 2.5s).
2. **Dynamic Interaction Radius:** For each scene, find the 90th percentile of the nearest neighbor distance across all global windows to serve as the dynamic interaction radius `r`.
3. **Graph Construction:** 
   - Build a `cKDTree` for all agents present in a global window based on their mean position.
   - Add nodes with velocity vector attributes (`vx, vy`).
   - Add undirected edges between agents within distance `r`. 
   - Cap maximum node degree to 8 globally.
   - Weight edges using Gaussian decay: $w = e^{-d^2 / (2 \sigma^2)}$ where $\sigma = r / 2$.
   - Save each graph as a `.gpickle` file.
4. **Extract Graph Features:** Calculate for each agent in the global window:
   - `nearest_neighbor_distance`, `relative_velocity_to_nearest`
   - `local_pedestrian_density` (agents in bounding box area)
   - `crowd_density` (total agents in global window)
   - `node_degree`, `node_centrality`
   - `interaction_change_rate` (Jaccard-like distance of neighbor sets between consecutive global windows).
5. **Robust Normalization:** Normalize the graph-derived density and distance features per scene using Median/IQR scaling (same as Part 2).
6. **Merge to Local Windows:** Join these global graph features back onto the individual local overlapping agent windows inside `features_windows.csv` and `node_features.csv`.

## 5. Finalize Weak Labels (`finalize_nus_weak.py`)
This stage recalculates the Navigation Uncertainty Score using the full set of kinematic and graph features and scales it for training.

**Input:** `features_windows_with_graph.csv`
**Output:** `features_final.csv`

### Steps:
1. **Include Graph Features:** Normalize `interaction_change_rate` (median/IQR).
2. **Recompute NUS_weak:** 
   - Use Expert Weights defined in the script to compute `NUS_weak_expert` as a linear combination of normalized features (e.g., stopping frequency, loop backtrack, turning variance, hesitation, path efficiency, interaction change rate).
   - Re-fit PCA on the active normalized features to generate `NUS_weak_pca`.
3. **Global Scaling:**
   - Find the global minimum and maximum of the computed `NUS_weak` scores.
   - Apply Min-Max scaling to map `NUS_weak_expert` and `NUS_weak_pca` to the range `[0, 1]`.
4. **Final Export:** Save all features and finalized target variables to `features_final.csv`, which is directly consumed by PyTorch Geometric dataset builders.

---

### Implementation Checklist for New Locations:
- [ ] Create `preprocess_<location>.py` and update the directory paths.
- [ ] Create `compute_features_<location>.py` and update paths.
- [ ] Create `compute_node_features_<location>.py` and update paths.
- [ ] Create `build_graphs_<location>.py` and update paths.
- [ ] Run scripts in numerical sequence.
- [ ] (Optional) Integrate dataset parsing in PyG loader (`build_pyg_dataset.py`).
