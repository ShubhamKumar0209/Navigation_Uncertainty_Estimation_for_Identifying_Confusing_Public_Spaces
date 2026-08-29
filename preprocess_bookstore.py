import os
import glob
import pandas as pd
import numpy as np

def load_and_parse_annotations(filepath: str) -> pd.DataFrame:
    """
    Loads and parses the annotations.txt file from the Stanford Drone Dataset.
    Columns: track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
    """
    columns = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    
    # Read CSV. It's space-separated
    df = pd.read_csv(filepath, sep=r'\s+', names=columns)
    
    # Cast to correct dtypes
    df['track_id'] = df['track_id'].astype(int)
    df['xmin'] = df['xmin'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    df['frame'] = df['frame'].astype(int)
    df['lost'] = df['lost'].astype(bool)
    df['occluded'] = df['occluded'].astype(bool)
    df['generated'] = df['generated'].astype(bool)
    df['label'] = df['label'].astype(str).str.replace('"', '').str.replace("'", "")
    
    return df

def filter_pedestrians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to keep only rows where label is 'Pedestrian'.
    Drops other classes like Biker, Cart, Car, Bus, Skater.
    """
    return df[df['label'] == "Pedestrian"].copy()

def drop_lost_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where the 'lost' flag is True (1).
    These are placeholder/out-of-frame coordinates, not real positions.
    """
    return df[~df['lost']].copy()

def compute_centroid_and_timestamp(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Computes the centroid position in pixels and the timestamp in seconds based on frame rate.
    """
    df['x_pixel'] = (df['xmin'] + df['xmax']) / 2.0
    df['y_pixel'] = (df['ymin'] + df['ymax']) / 2.0
    df['timestamp'] = df['frame'] / fps
    return df

def apply_homography(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for applying a homography transformation from pixel coordinates to world coordinates.
    Currently returns the dataframe unchanged, assigning pixel coordinates directly to world coordinates.
    """
    # TODO: homography - apply actual pixel-to-metric homography here
    df['x_world'] = df['x_pixel']
    df['y_world'] = df['y_pixel']
    return df

def build_unified_schema(df: pd.DataFrame, scene_id: str) -> pd.DataFrame:
    """
    Transforms the dataframe into the unified schema:
    [dataset, scene_id, agent_id, frame, timestamp, x_world, y_world, agent_type]
    """
    df = apply_homography(df)
    
    unified_df = pd.DataFrame()
    unified_df['dataset'] = ["SDD"] * len(df)
    unified_df['scene_id'] = [scene_id] * len(df)
    unified_df['agent_id'] = df['track_id'].astype(str)
    unified_df['frame'] = df['frame']
    unified_df['timestamp'] = df['timestamp']
    unified_df['x_world'] = df['x_world']
    unified_df['y_world'] = df['y_world']
    unified_df['agent_type'] = ["pedestrian"] * len(df)
    
    return unified_df

def split_tracks_by_gap(df: pd.DataFrame, max_gap_sec: float = 1.0) -> pd.DataFrame:
    """
    Splits tracks that have consecutive observations with a time gap greater than `max_gap_sec`.
    New tracks are assigned synthetic agent_ids (e.g., '{original_id}_b').
    """
    if len(df) == 0:
        return df
        
    # Sort by agent_id and timestamp
    df = df.sort_values(['agent_id', 'timestamp']).copy()
    
    # Calculate time difference between consecutive rows within the same agent
    df['time_diff'] = df.groupby('agent_id')['timestamp'].diff()
    
    # Identify where a new split should happen (time gap > max_gap_sec)
    df['split_flag'] = (df['time_diff'] > max_gap_sec).astype(int)
    
    # Cumulative sum gives a split index (0, 1, 2...) for each agent
    df['split_idx'] = df.groupby('agent_id')['split_flag'].cumsum()
    
    def generate_new_id(row):
        base_id = row['agent_id']
        idx = row['split_idx']
        if pd.isna(idx):
            idx = 0
        idx = int(idx)
        if idx == 0:
            return base_id
        else:
            # Generate a suffix based on the split index (1 -> b, 2 -> c, etc.)
            suffix = chr(ord('a') + idx)
            return f"{base_id}_{suffix}"
            
    df['agent_id'] = df.apply(generate_new_id, axis=1)
    
    # Clean up temporary columns
    df = df.drop(columns=['time_diff', 'split_flag', 'split_idx'])
    
    return df

def resample_and_filter_tracks(df: pd.DataFrame, dt: float = 0.4, min_duration: float = 5.0) -> pd.DataFrame:
    """
    Resamples tracks to a fixed timestep `dt` using linear interpolation and 
    discards tracks shorter than `min_duration`.
    """
    resampled_rows = []
    
    # Group by the potentially split agent_ids
    for agent_id, track in df.groupby('agent_id'):
        min_time = track['timestamp'].min()
        max_time = track['timestamp'].max()
        
        # Check track duration
        if (max_time - min_time) < min_duration:
            continue
            
        # Create a fixed time grid for resampling
        # Starting from the track's min_time to ensure we capture the full track bounds appropriately
        new_times = np.arange(min_time, max_time + 1e-9, dt)
        
        # Interpolate x and y
        new_x = np.interp(new_times, track['timestamp'], track['x_world'])
        new_y = np.interp(new_times, track['timestamp'], track['y_world'])
        new_frames = np.interp(new_times, track['timestamp'], track['frame']).round().astype(int)
        
        # Build the new track dataframe
        first_row = track.iloc[0]
        new_track = pd.DataFrame({
            'dataset': first_row['dataset'],
            'scene_id': first_row['scene_id'],
            'agent_id': agent_id,
            'frame': new_frames,
            'timestamp': new_times,
            'x_world': new_x,
            'y_world': new_y,
            'agent_type': first_row['agent_type']
        })
        resampled_rows.append(new_track)
        
    if not resampled_rows:
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(resampled_rows, ignore_index=True)

def process_video(video_path: str, output_dir: str) -> dict:
    """
    Processes a single video directory and saves the CSV file.
    Returns a dictionary of statistics for the sanity check summary.
    """
    scene_id = f"bookstore_{os.path.basename(video_path)}"
    annotations_file = os.path.join(video_path, 'annotations.txt')
    
    if not os.path.exists(annotations_file):
        print(f"Warning: {annotations_file} not found.")
        return None
        
    # 1. Load and parse
    df = load_and_parse_annotations(annotations_file)
    total_objects_before = df['track_id'].nunique()
    
    # 2. Filter pedestrians
    df = filter_pedestrians(df)
    pedestrians_before = df['track_id'].nunique()
    
    if len(df) == 0:
        return None
        
    # 3. Drop lost
    df = drop_lost_annotations(df)
    
    # 4 & 5. Compute centroid and timestamp
    df = compute_centroid_and_timestamp(df)
    
    # 6. Build unified schema
    df = build_unified_schema(df, scene_id)
    
    # Track split stats
    original_track_count = df['agent_id'].nunique()
    
    # 7. Split tracks by gap
    df = split_tracks_by_gap(df)
    split_track_count = df['agent_id'].nunique()
    tracks_split = split_track_count - original_track_count
    
    # 8. Resample and discard tiny tracks
    df = resample_and_filter_tracks(df)
    final_pedestrians = df['agent_id'].nunique()
    
    if len(df) > 0:
        avg_track_length = df.groupby('agent_id')['timestamp'].apply(lambda x: x.max() - x.min()).mean()
    else:
        avg_track_length = 0.0
        
    # 9. Save output
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{scene_id}.csv")
    df.to_csv(out_file, index=False)
    
    stats = {
        'scene_id': scene_id,
        'total_objects_raw': total_objects_before,
        'pedestrians_raw': pedestrians_before,
        'final_tracks': final_pedestrians,
        'tracks_split': tracks_split,
        'avg_track_length_sec': avg_track_length,
        'df': df
    }
    
    return stats

def main():
    base_dir = "bookstore"
    output_dir = os.path.join("processed", "bookstore")
    
    video_folders = sorted(glob.glob(os.path.join(base_dir, "video*")))
    if not video_folders:
        print(f"No video folders found in {base_dir}")
        return
        
    all_dfs = []
    
    print("Processing videos...")
    print("-" * 85)
    print(f"{'Scene':<18} | {'Raw Objs':<10} | {'Raw Peds':<10} | {'Splits':<8} | {'Final Tracks':<12} | {'Avg Len (s)':<12}")
    print("-" * 85)
    
    for folder in video_folders:
        stats = process_video(folder, output_dir)
        if stats:
            print(f"{stats['scene_id']:<18} | {stats['total_objects_raw']:<10} | {stats['pedestrians_raw']:<10} | {stats['tracks_split']:<8} | {stats['final_tracks']:<12} | {stats['avg_track_length_sec']:<12.2f}")
            if len(stats['df']) > 0:
                all_dfs.append(stats['df'])
                
    print("-" * 85)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_out = os.path.join(output_dir, "bookstore_all.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"\nSaved combined CSV file to {combined_out}")
        # agent_id is only unique per scene, so we group by scene_id and agent_id
        total_unique = combined_df.groupby(['scene_id', 'agent_id']).ngroups
        print(f"Total final pedestrian tracks across all scenes: {total_unique}")
    else:
        print("\nNo valid tracks found to combine.")

if __name__ == "__main__":
    main()
