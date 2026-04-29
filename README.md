# PIMSR

PIMSR is a PyTorch implementation of a physics-informed multimodal super-resolution framework for digital elevation model (DEM) reconstruction. The model uses low-resolution DEM input together with co-registered Sentinel-1 VV/VH SAR observations and incidence angle maps to recover high-resolution DEM patches.

## Features

- SAR-guided DEM super-resolution network
- Differentiable SAR simulator for physics-informed supervision
- HDF5-based dataset loading for efficient training and evaluation
- Training, validation, and testing scripts
- Metric reporting for PSNR, SSIM, RMSE, and MAE
- GeoTIFF export support during testing
- Data preparation utilities for HDF5 packing, LR tile generation, and statistics computation

## Repository Structure

```text
.
├── config.py
├── data_preparation/
│   ├── calculate_global_stats.py
│   ├── calculate_global_stats_dem.py
│   ├── create_lr_tiles.py
│   └── preprocess_to_hdf5.py
├── dataset.py
├── logger.py
├── metrics.py
├── networks.py
├── sar_Simulator.py
├── sar_enhanced_model.py
├── test.py
├── train.py
├── trainer.py
└── visualization.py
```

## Environment

Recommended dependencies:

- Python 3.10 or later
- PyTorch
- NumPy
- Pandas
- Matplotlib
- tqdm
- h5py
- scikit-image
- GDAL
- rasterio

Example Conda setup:

```bash
conda create -n pimsr python=3.10
conda activate pimsr

conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge numpy pandas matplotlib tqdm h5py scikit-image gdal rasterio
```

Install the PyTorch build that matches your CUDA version if GPU training is required.

## Data Format

### HDF5 Files

Each HDF5 file should contain the following datasets:

```text
hr_dem_raw     (N, 64, 64)
lr_dem_raw     (N, 16, 16)
vv_sar_raw     (N, 64, 64)
vh_sar_raw     (N, 64, 64)
inc_angle_map  (N, 64, 64)
```

Dataset meanings:

- `hr_dem_raw`: ground-truth high-resolution DEM patch
- `lr_dem_raw`: low-resolution DEM input patch
- `vv_sar_raw`: co-registered VV SAR patch
- `vh_sar_raw`: co-registered VH SAR patch
- `inc_angle_map`: incidence angle map aligned with the patch

### CSV Index Files

The HDF5 loader reads arrays directly from the HDF5 files. CSV index files are used for sample bookkeeping, consistency checks, and GeoTIFF filename lookup during testing.

Expected CSV format:

```csv
patch_id,dem_path,lr_path,vv_path,vh_path,angle_path
patch_000001,DEM/patch_000001.tif,LR/patch_000001.tif,VV/patch_000001.tif,VH/patch_000001.tif,Angle/patch_000001.tif
patch_000002,DEM/patch_000002.tif,LR/patch_000002.tif,VV/patch_000002.tif,VH/patch_000002.tif,Angle/patch_000002.tif
```

The row order in the CSV file should match the sample order in the corresponding HDF5 file.

### Global Statistics

Training and testing expect a JSON statistics file with this structure:

```json
{
  "dem": {
    "mean": 0.0,
    "std": 1.0
  },
  "vv_log": {
    "mean": 0.0,
    "std": 1.0
  },
  "vh_log": {
    "mean": 0.0,
    "std": 1.0
  }
}
```

`vv_log` and `vh_log` are computed from raw SAR values after applying:

```text
log(1 + x)
```

## Recommended Data Layout

```text
data/
├── Train_data/
│   ├── train_data_real_lr.h5
│   ├── data_index.csv
│   └── global_stats.json
├── Val_data/
│   ├── val_data_real_lr.h5
│   └── data_index.csv
└── Test_data/
    ├── test_data_real_lr.h5
    ├── data_index.csv
    └── tiles/
```

## Data Preparation

The `data_preparation/` directory provides utilities for building the required dataset files.

Typical workflow:

1. Prepare spatially aligned HR DEM, LR DEM, VV SAR, VH SAR, and incidence angle GeoTIFF patches.
2. Create CSV index files with relative paths to each patch.
3. Pack the patch arrays into HDF5 files using the required dataset keys.
4. Compute dataset-level statistics and save them as `global_stats.json`.

Available scripts:

- `data_preparation/create_lr_tiles.py`: create aligned LR DEM tiles from a large low-resolution raster
- `data_preparation/preprocess_to_hdf5.py`: pack GeoTIFF tiles into HDF5 files
- `data_preparation/calculate_global_stats.py`: compute statistics for DEM, VV SAR, and VH SAR
- `data_preparation/calculate_global_stats_dem.py`: compute DEM-only statistics

## Training

Run training with `train.py`:

```bash
python train.py \
  --train_h5_path /path/to/Train_data/train_data_real_lr.h5 \
  --train_csv_path /path/to/Train_data/data_index.csv \
  --val_h5_path /path/to/Val_data/val_data_real_lr.h5 \
  --val_csv_path /path/to/Val_data/data_index.csv \
  --global_stats /path/to/Train_data/global_stats.json \
  --simulator_weights /path/to/best_precise_simulator.pth \
  --output_dir /path/to/output/train_run \
  --batch_size 16 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --num_workers 4 \
  --lr_step_size 25 \
  --lr_gamma 0.5 \
  --recon_weight 1.0 \
  --sar_weight 0.1 \
  --device cuda \
  --random_seed 1008
```

Training outputs are written to:

```text
output_dir/
├── checkpoints/
├── logs/
└── visualizations/
```

The visualizations directory includes sample prediction figures and the training curve plot generated after training completes.

## Testing

Run evaluation with `test.py`:

```bash
python test.py \
  --test_h5_path /path/to/Test_data/test_data_real_lr.h5 \
  --test_dataset_csv /path/to/Test_data/data_index.csv \
  --global_stats /path/to/Train_data/global_stats.json \
  --model_weights /path/to/output/train_run/checkpoints/best_model.pth \
  --output_dir /path/to/output/test_run \
  --test_data_root /path/to/Test_data/tiles \
  --batch_size 16 \
  --num_workers 4 \
  --vis_freq 10
```

The test script reports PSNR, SSIM, RMSE, and MAE for the model output and the bicubic baseline. When GeoTIFF export is enabled, geospatial metadata is copied from the source DEM files identified by the CSV index.

## SAR Simulator

The differentiable SAR simulator is implemented in `sar_Simulator.py`. During full physics-informed training, the super-resolution model uses a SAR consistency loss computed with a pretrained simulator checkpoint:

```bash
--simulator_weights /path/to/best_precise_simulator.pth
```

If a pretrained simulator checkpoint is not available, prepare or train one before enabling the SAR consistency loss.

## Minimal Workflow

1. Prepare aligned GeoTIFF patches.
2. Create CSV index files.
3. Pack train, validation, and test sets into HDF5 files.
4. Compute `global_stats.json`.
5. Prepare a SAR simulator checkpoint.
6. Train the DEM super-resolution model with `train.py`.
7. Evaluate the trained model with `test.py`.

