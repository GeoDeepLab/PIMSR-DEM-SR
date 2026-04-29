import os
import numpy as np
import pandas as pd
from osgeo import gdal
import rasterio
from pathlib import Path
from tqdm import tqdm
import json

def calculate_and_save_stats(root_dir, csv_file, output_path):
    """Compute global mean and standard deviation for all data streams.

    Save the results as a JSON file.

    Args:
        root_dir (Path): Root directory containing all patch folders.
        csv_file (str): Dataset index CSV file name.
        output_path (Path): Output path for the statistics JSON file.
    """

    if output_path.exists():
        print(f"The statistics file already exists at: {output_path}")
        user_input = input("Do you want to recompute it? (y/n): ").lower()
        if user_input != 'y':
            print("Operation cancelled.")
            return

    print("Start calculating the global normalized statistics of the dataset")
    df = pd.read_csv(root_dir / csv_file)
    
    # Use generators to reduce memory usage.
    def dem_gen():
        for _, row in df.iterrows():
            with rasterio.open(root_dir / row['dem_path']) as src:
                yield src.read(1).astype(np.float32).flatten()
    def vv_log_gen():
        for _, row in df.iterrows():
            with rasterio.open(root_dir / row['vv_path']) as src:
                yield np.log1p(src.read(1).astype(np.float32)).flatten()
    def vh_log_gen():
        for _, row in df.iterrows():
            with rasterio.open(root_dir / row['vh_path']) as src:
                yield np.log1p(src.read(1).astype(np.float32)).flatten()

    # Use np.concatenate and tqdm to process large data streams.
    dem_all = np.concatenate(list(tqdm(dem_gen(), total=len(df), desc="Processing DEM")))
    vv_log_all = np.concatenate(list(tqdm(vv_log_gen(), total=len(df), desc="Processing VV")))
    vh_log_all = np.concatenate(list(tqdm(vh_log_gen(), total=len(df), desc="Processing VH")))
    
    stats = {
        'dem': {'mean': float(np.mean(dem_all)), 'std': float(np.std(dem_all))},
        'vv_log': {'mean': float(np.mean(vv_log_all)), 'std': float(np.std(vv_log_all))},
        'vh_log': {'mean': float(np.mean(vh_log_all)), 'std': float(np.std(vh_log_all))},
    }
    
    # Save as a JSON file.
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\nStatistical data calculation complete.")
    print(f"Global statistics have been saved to:{output_path}")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    TRAIN_DATA_ROOT = Path(r"path/to/your/Train_data")

    TRAIN_CSV = "data_index.csv"
    STATS_JSON_PATH = TRAIN_DATA_ROOT / "global_stats.json"
    
    calculate_and_save_stats(TRAIN_DATA_ROOT, TRAIN_CSV, STATS_JSON_PATH)
