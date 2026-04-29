import rasterio
import cv2
from pathlib import Path
from tqdm import tqdm
from rasterio.windows import from_bounds


# Directory containing HR DEM GeoTIFF tiles.
HR_TILES_DIR = Path(r"path/to/your/HR_DEM_tiles")

# Full path to the LR DEM mosaic GeoTIFF file.
LR_MASTER_GEOTIFF_PATH = Path(
    r"path/to/your/LR_DEM_mosaic/data"
)

# Output directory for cropped LR DEM tiles.
OUTPUT_LR_TILES_DIR = Path(r"path/to/your/cropped_LR_DEM_tiles")


# Target LR tile size.
TARGET_LR_SIZE = (16, 16)  # Width, height.


def create_corresponding_lr_tiles():
    """Crop a corresponding LR tile from the LR master raster for each HR tile."""
    print("--- Starting to create corresponding low-resolution (LR) tiles ---")

    # Ensure the output directory exists.
    OUTPUT_LR_TILES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_LR_TILES_DIR}")

    # Find all HR tile files.
    hr_files = list(HR_TILES_DIR.glob('*.tif')) + list(HR_TILES_DIR.glob('*.tiff'))
    if not hr_files:
        print(f"[Error] No .tif or .tiff files were found in: {HR_TILES_DIR}")
        print("Please check the input directory.")
        return

    print(f"Found {len(hr_files)} HR tile files.")

    # Open the LR master raster once.
    try:
        with rasterio.open(LR_MASTER_GEOTIFF_PATH) as lr_master_src:
            print(f"Successfully opened the LR master file: {LR_MASTER_GEOTIFF_PATH.name}")

            # Check and report CRS information.
            with rasterio.open(hr_files[0]) as hr_sample_src:
                hr_sample_crs = hr_sample_src.crs

            print(f"HR tile CRS: {hr_sample_crs}")
            print(f"LR master file CRS: {lr_master_src.crs}")

            if lr_master_src.crs != hr_sample_crs:
                print("\nWarning: The CRS of the HR tiles and the LR master file do not match.")

            # Create a tqdm progress bar.
            for hr_path in tqdm(hr_files, desc="Processing tiles"):
                try:
                    # Open the HR tile to get its geographic bounds.
                    with rasterio.open(hr_path) as hr_src:
                        bounds = hr_src.bounds

                    # Use HR bounds to compute the corresponding pixel window in the LR raster.
                    window = from_bounds(*bounds, lr_master_src.transform)

                    # Read the window data from the LR master raster.
                    lr_data = lr_master_src.read(
                        1,
                        window=window,
                        boundless=True,
                        fill_value=0
                    )

                    # Resize the cropped data to the target tile size.
                    lr_data_resized = cv2.resize(
                        lr_data,
                        TARGET_LR_SIZE,
                        interpolation=cv2.INTER_CUBIC
                    )

                    # Prepare metadata for the output file.
                    output_profile = lr_master_src.profile.copy()

                    # Compute the geotransform for the new tile.
                    output_transform = rasterio.windows.transform(
                        window,
                        lr_master_src.transform
                    )

                    output_profile.update({
                        'height': TARGET_LR_SIZE[1],
                        'width': TARGET_LR_SIZE[0],
                        'transform': output_transform,
                        'compress': 'lzw'
                    })

                    # Define the output path with the same filename as the HR tile.
                    output_path = OUTPUT_LR_TILES_DIR / hr_path.name

                    # Write the output file.
                    with rasterio.open(output_path, 'w', **output_profile) as dst:
                        dst.write(lr_data_resized.astype(output_profile['dtype']), 1)

                except Exception as e:
                    print(f"\nError processing file {hr_path.name}: {e}")
                    continue

    except rasterio.errors.RasterioIOError:
        print(f"Failed to open the LR master file: {LR_MASTER_GEOTIFF_PATH}")
        print("Please check whether the file exists and is not corrupted.")
        return

    print(f"All cropped LR tiles have been saved to: {OUTPUT_LR_TILES_DIR}")


if __name__ == '__main__':
    create_corresponding_lr_tiles()