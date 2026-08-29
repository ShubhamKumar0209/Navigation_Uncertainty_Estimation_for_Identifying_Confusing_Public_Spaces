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
