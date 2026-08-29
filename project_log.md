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
