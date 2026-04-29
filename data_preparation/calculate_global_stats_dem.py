import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import json


def calculate_dem_stats(root_dir, csv_file, output_path):
    """Compute global mean and standard deviation for DEM data only.

    Save the results as a JSON file with the expected format.

    Args:
        root_dir (Path): Root directory containing all data folders.
        csv_file (str): Dataset index CSV file name; it should contain a 'dem_path' column.
        output_path (Path): Output path for the statistics JSON file.
    """
    if output_path.exists():
        print(f"The statistics file already exists at: {output_path}")
        user_input = input("Do you want to recompute it? (y/n): ").lower()
        if user_input != 'y':
            print("Operation cancelled.")
            return

    print("Starting to compute global normalization statistics for the DEM dataset.")

    try:
        df = pd.read_csv(root_dir / csv_file)
        if 'dem_path' not in df.columns:
            print(f"The column 'dem_path' was not found in the CSV file: {csv_file}")
            return
    except FileNotFoundError:
        print(f"CSV file not found: {root_dir / csv_file}")
        return

    def dem_generator():
        for _, row in df.iterrows():
            dem_path = root_dir / row['dem_path']
            if not dem_path.exists():
                print(f"File not found: {dem_path}. Skipping.")
                continue

            try:
                with rasterio.open(dem_path) as src:
                    # Read the first band, convert it to float32, and flatten it.
                    yield src.read(1).astype(np.float32).flatten()
            except Exception as e:
                print(f"Error reading file {dem_path}: {e}. Skipping.")

    try:
        dem_all_pixels = np.concatenate(
            list(tqdm(dem_generator(), total=len(df), desc="Processing DEM"))
        )
    except ValueError:
        print("\nNo DEM data was successfully read, so the statistics cannot be computed.")
        print("Please check the file paths and file contents.")
        return

    dem_mean = np.mean(dem_all_pixels)
    dem_std = np.std(dem_all_pixels)

    stats = {
        'dem': {
            'mean': float(dem_mean),
            'std': float(dem_std)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print("\nStatistics computation completed.")
    print(f"Global statistics have been saved to: {output_path}")
    print("\nComputed results:")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':

    DATA_ROOT = Path(r"path/to/your/Train_data")

    DATA_CSV = "data_index_cleaned.csv"

    STATS_JSON_PATH = DATA_ROOT / "dem_global_stats.json"

    calculate_dem_stats(DATA_ROOT, DATA_CSV, STATS_JSON_PATH)
