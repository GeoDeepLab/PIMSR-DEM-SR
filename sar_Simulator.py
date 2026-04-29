
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
import pandas as pd
from osgeo import gdal
import rasterio
import numpy as np

from tqdm import tqdm
import time
import random
import json
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


class PretrainDatasetWithAngleMap(Dataset):
    def __init__(self, csv_file, root_dir, normalization_stats):
        self.patch_info_df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = Path(root_dir)
        self.stats = normalization_stats

    def __len__(self):
        return len(self.patch_info_df)

    def __getitem__(self, idx):
        patch_row = self.patch_info_df.iloc[idx]
        
        # Read DEM, VV, and VH.
        with rasterio.open(self.root_dir / patch_row['dem_path']) as src: 
            dem_data = src.read(1).astype(np.float32)
        with rasterio.open(self.root_dir / patch_row['vv_path']) as src: 
            vv_data = src.read(1).astype(np.float32)
        with rasterio.open(self.root_dir / patch_row['vh_path']) as src: 
            vh_data = src.read(1).astype(np.float32)
            
        with rasterio.open(self.root_dir / patch_row['angle_path']) as src: 
            inc_angle_map_data = src.read(1).astype(np.float32)
        # Apply log transform and normalization to raw SAR data.
        vv_log = np.log1p(vv_data)
        vh_log = np.log1p(vh_data)
        
        vv_norm = (vv_log - self.stats['vv_log']['mean']) / self.stats['vv_log']['std']
        vh_norm = (vh_log - self.stats['vh_log']['mean']) / self.stats['vh_log']['std']
        
        sar_target_norm = np.stack([vv_norm, vh_norm], axis=0)
        
        return {
            'dem_true': torch.from_numpy(dem_data).unsqueeze(0),
            'sar_target_norm': torch.from_numpy(sar_target_norm),
            'inc_angle_map': torch.from_numpy(inc_angle_map_data).unsqueeze(0)
        }

class PreciseAngleSimulator(nn.Module):
    """
    """
    def __init__(self, correction_scale_deg=30.0): 
        super().__init__()
        
        self.log_A_vv = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_K_vv = nn.Parameter(torch.log(torch.tensor(4.0)))
        self.log_A_vh = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_K_vh = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.log_C_diffuse = nn.Parameter(torch.log(torch.tensor(0.05)))
        
        self.dem_to_angle_correction = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1, bias=False),
            nn.Tanh()  # Output range is [-1, 1].
        )
        self.correction_scale = correction_scale_deg # Map Tanh output from [-1, 1] to [-scale, +scale] degrees.

    def forward(self, dem_input, true_inc_angle_map_deg):
        angle_correction_deg = self.dem_to_angle_correction(dem_input) * self.correction_scale
        
        final_inc_angle_deg = F.relu(true_inc_angle_map_deg + angle_correction_deg)
        
        cos_theta_loc = torch.cos(torch.deg2rad(final_inc_angle_deg))
        
        A_vv = torch.exp(self.log_A_vv)
        K_vv = torch.exp(self.log_K_vv)
        A_vh = torch.exp(self.log_A_vh)
        K_vh = torch.exp(self.log_K_vh)
        C_diffuse = torch.exp(self.log_C_diffuse)
        
        diffuse_component = C_diffuse * cos_theta_loc
        specular_component_vv = A_vv * (cos_theta_loc.pow(K_vv))
        sar_vv_sim = specular_component_vv + diffuse_component
        volume_component_vh = A_vh * (cos_theta_loc.pow(K_vh))
        sar_vh_sim = volume_component_vh
        
        return torch.cat([sar_vv_sim, sar_vh_sim], dim=1)

class L1_GDL_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        sobel_x_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
        sobel_y_kernel = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
        self.sobel_x = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False, groups=2)
        self.sobel_y = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False, groups=2)
        self.sobel_x.weight.data = torch.cat([sobel_x_kernel, sobel_x_kernel], dim=0)
        self.sobel_y.weight.data = torch.cat([sobel_y_kernel, sobel_y_kernel], dim=0)
        for p in self.sobel_x.parameters(): p.requires_grad = False
        for p in self.sobel_y.parameters(): p.requires_grad = False

    def forward(self, y_pred, y_true):
        self.sobel_x.to(y_pred.device)
        self.sobel_y.to(y_pred.device)
        loss_l1 = self.l1(y_pred, y_true)
        pred_grad_x = self.sobel_x(y_pred); pred_grad_y = self.sobel_y(y_pred)
        true_grad_x = self.sobel_x(y_true); true_grad_y = self.sobel_y(y_true)
        loss_gdl = self.l1(pred_grad_x, true_grad_x) + self.l1(pred_grad_y, true_grad_y)
        return (1 - self.alpha) * loss_l1 + self.alpha * loss_gdl

def visualize_precise_results(epoch, model, vis_dataloader, device, output_dir, stats):
    model.eval()
    plt.ioff()
    with torch.no_grad():
        batch = next(iter(vis_dataloader))
        dem_true = batch['dem_true'].to(device)
        sar_target_norm = batch['sar_target_norm'].to(device)
        inc_angle_map = batch['inc_angle_map'].to(device)

        sar_sim_raw = model(dem_true, inc_angle_map)

        sar_sim_log = torch.log1p(sar_sim_raw)
        vv_sim_norm = (sar_sim_log[:, 0:1, :, :] - stats['vv_log']['mean']) / stats['vv_log']['std']
        vh_sim_norm = (sar_sim_log[:, 1:2, :, :] - stats['vh_log']['mean']) / stats['vh_log']['std']
        sar_sim_norm = torch.cat([vv_sim_norm, vh_sim_norm], dim=1)
        
        VIS_SAMPLE_COUNT = min(4, dem_true.shape[0])
        fig, axes = plt.subplots(VIS_SAMPLE_COUNT, 5, figsize=(20, 4 * VIS_SAMPLE_COUNT), squeeze=False)
        fig.suptitle(f'Epoch {epoch+1} - Precise local_Angle Simulator', fontsize=16)
        
        for i in range(VIS_SAMPLE_COUNT):
            axes[i, 0].imshow(sar_target_norm[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1} - Target VV')
            
            axes[i, 1].imshow(sar_sim_norm[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('Simulated VV')
            
            axes[i, 2].imshow(dem_true[i, 0].cpu().numpy(), cmap='terrain')
            axes[i, 2].set_title('Source DEM')
            
            im = axes[i, 3].imshow(inc_angle_map[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 3].set_title('Inc. local_Angle Map (°)')
            plt.colorbar(im, ax=axes[i, 3])

            im = axes[i, 4].imshow((sar_target_norm[i, 0] - sar_sim_norm[i, 0]).cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            axes[i, 4].set_title('Error Map (Target - Sim)')
            plt.colorbar(im, ax=axes[i, 4])

            for ax in axes[i]: 
                ax.set_xticks([]); ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        vis_path = output_dir / f"precise_visualization_epoch_{epoch+1:03d}.png"
        plt.savefig(vis_path, dpi=150)
        plt.close(fig)
    print(f"  - Visualization for epoch {epoch+1} saved.")

def calculate_original_stats(root_dir, csv_file):
    print("Calculating global normalized statistics for the raw SAR...")
    df = pd.read_csv(os.path.join(root_dir, csv_file))

    def vv_log_gen():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating VV stats"):
            with rasterio.open(root_dir / row['vv_path']) as src:
                vv_data = src.read(1).astype(np.float32)
                yield np.log1p(vv_data).flatten()
    
    def vh_log_gen():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating VH stats"):
            with rasterio.open(root_dir / row['vh_path']) as src:
                vh_data = src.read(1).astype(np.float32)
                yield np.log1p(vh_data).flatten()

    vv_log_all = np.concatenate(list(vv_log_gen()))
    vh_log_all = np.concatenate(list(vh_log_gen()))
    
    stats = {
        'vv_log': {'mean': float(np.mean(vv_log_all)), 'std': float(np.std(vv_log_all))},
        'vh_log': {'mean': float(np.mean(vh_log_all)), 'std': float(np.std(vh_log_all))},
    }
    print("Statistical data calculation completed.")
    return stats


def pretrain_precise_simulator():
    ROOT_DIR = Path(r"path/to/your/Train_data")
    CSV_FILE = "data_index.csv"
    OUTPUT_DIR = Path(r"path/to/your/precise_simulator_output")
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    L1_GDL_ALPHA = 0.3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = str(OUTPUT_DIR / "training_log.csv")
    STATS_FILE = str(OUTPUT_DIR / "original_norm_stats.json")
    
    print("="*60 + f"\nStarting Precise local_Angle Simulator Training at {time.ctime()}\n" + "="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if os.path.exists(STATS_FILE):
        print(f"Loading normalization stats from {STATS_FILE}...")
        with open(STATS_FILE, 'r') as f: norm_stats = json.load(f)
    else:
        norm_stats = calculate_original_stats(ROOT_DIR, CSV_FILE)
        with open(STATS_FILE, 'w') as f: json.dump(norm_stats, f, indent=4)
        print(f"Normalization stats saved to {STATS_FILE}")
    print(f"Normalization Stats: {json.dumps(norm_stats, indent=2)}")
        
    dataset = PretrainDatasetWithAngleMap(csv_file=CSV_FILE, root_dir=ROOT_DIR, 
                                          normalization_stats=norm_stats)

    vis_dataloader = DataLoader(Subset(dataset, random.sample(range(len(dataset)), 4)), batch_size=4, shuffle=False)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=True)
    
    model = PreciseAngleSimulator().to(device)
    criterion = L1_GDL_Loss(alpha=L1_GDL_ALPHA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    log_header = "Epoch,Total_Loss,PSNR,SSIM,LR,A_vv,K_vv,C_diffuse,A_vh,K_vh,Duration_s\n"
    with open(LOG_FILE, 'w') as f: f.write(log_header)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        epoch_loss, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            dem_true = batch['dem_true'].to(device)
            sar_target_norm = batch['sar_target_norm'].to(device)
            inc_angle_map = batch['inc_angle_map'].to(device)
            
            optimizer.zero_grad()
            
            sar_sim_raw = model(dem_true, inc_angle_map)
            
            sar_sim_log = torch.log1p(sar_sim_raw)
            vv_sim_norm = (sar_sim_log[:, 0:1] - norm_stats['vv_log']['mean']) / norm_stats['vv_log']['std']
            vh_sim_norm = (sar_sim_log[:, 1:2] - norm_stats['vh_log']['mean']) / norm_stats['vh_log']['std']
            sar_sim_norm = torch.cat([vv_sim_norm, vh_sim_norm], dim=1)

            loss = criterion(sar_sim_norm, sar_target_norm)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            with torch.no_grad():
                sar_sim_np = sar_sim_norm.cpu().numpy()
                sar_true_np = sar_target_norm.cpu().numpy()
                data_range = sar_true_np.max() - sar_true_np.min()
                if data_range < 1e-6: data_range = 1.0
                epoch_psnr += psnr_metric(sar_true_np, sar_sim_np, data_range=data_range)
                epoch_ssim += ssim_metric(sar_true_np, sar_sim_np, channel_axis=1, data_range=data_range)

        avg_loss = epoch_loss / len(train_dataloader)
        avg_psnr = epoch_psnr / len(train_dataloader)
        avg_ssim = epoch_ssim / len(train_dataloader)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        duration = time.time() - epoch_start_time
        
        phys_params = {k: v.item() for k, v in model.named_parameters() if 'log' in k}

        print(
            f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f}dB | "
            f"SSIM: {avg_ssim:.4f} | LR: {current_lr:.1e} | "
            f"A_vv: {np.exp(phys_params['log_A_vv']):.3f}"
        )
        
        # Append the training log.
        with open(LOG_FILE, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_psnr:.4f},{avg_ssim:.4f},{current_lr:.4e},"
                    f"{np.exp(phys_params['log_A_vv']):.4f},{np.exp(phys_params['log_K_vv']):.4f},"
                    f"{np.exp(phys_params['log_C_diffuse']):.4f},{np.exp(phys_params['log_A_vh']):.4f},"
                    f"{np.exp(phys_params['log_K_vh']):.4f},{duration:.2f}\n")
        
        if (epoch + 1) % 1 == 0 or epoch == 0 or (epoch + 1) == NUM_EPOCHS:
            visualize_precise_results(epoch, model, vis_dataloader, device, OUTPUT_DIR, norm_stats)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_precise_simulator.pth")
            print(f"  - New best model saved with loss: {best_loss:.6f}")

    print(f"\nPrecise simulator training finished. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    pretrain_precise_simulator()
