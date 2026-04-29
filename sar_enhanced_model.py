import torch
import torch.nn as nn
import torch.nn.functional as F
from sar_Simulator import PreciseAngleSimulator

def swish(x):
    return F.relu(x)

# Channel attention module.
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial attention module.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        attention_weights = self.sigmoid(out)
        return attention_weights

# CBAM attention module.
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        self.save_attention = False
        self.spatial_weights = None
        self.channel_weights = None
    
    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        if self.save_attention:
            self.channel_weights = channel_att.detach()
            self.spatial_weights = spatial_att.detach()
        
        return x

# Residual block with CBAM attention.
class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.cbam = CBAM(n)
    
    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.cbam(y)
        return y + x


# DEM feature extraction module.
class DEMFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEMFeatureExtractor, self).__init__()
        # Gradient feature extraction.
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize gradient operators.
        vertical_kernel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        horizontal_kernel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        diagonal_kernel = torch.FloatTensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            self.conv_v.weight.data.copy_(vertical_kernel.repeat(out_channels, in_channels, 1, 1))
            self.conv_h.weight.data.copy_(horizontal_kernel.repeat(out_channels, in_channels, 1, 1))
            self.conv_d.weight.data.copy_(diagonal_kernel.repeat(out_channels, in_channels, 1, 1))
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        grad_v = self.conv_v(x)
        grad_h = self.conv_h(x)
        grad_d = self.conv_d(x)
        
        grad_features = torch.cat([grad_v, grad_h, grad_d], dim=1)
        fused_features = self.fusion(grad_features)
        fused_features = self.activation(fused_features)
        enhanced_features = self.cbam(fused_features)
        
        return enhanced_features


# SAR loss.
class SARLoss(nn.Module):
    def __init__(self, simulator_weights=None):
        super(SARLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # Load the SAR simulator.
        if simulator_weights:
            self.sar_simulator = PreciseAngleSimulator()
            try:
                # Try to load weights.
                checkpoint = torch.load(simulator_weights, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Load from a checkpoint dictionary with model_state_dict.
                    self.sar_simulator.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Load directly from a state_dict.
                    self.sar_simulator.load_state_dict(checkpoint)
                print(f"Loaded SAR simulator from {simulator_weights}")
                self.sar_simulator.eval()  # Set evaluation mode.
                # Freeze SAR simulator parameters.
                for param in self.sar_simulator.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Error loading SAR simulator weights: {e}")
                self.sar_simulator = None
        else:
            self.sar_simulator = None
    
    def forward(self, sr_dem, hr_dem, incidence_angle):
        """
        Compute the SAR consistency loss.
        
        Args:
            sr_dem: Generated high-resolution DEM.
            hr_dem: Ground-truth high-resolution DEM.
            incidence_angle: Incidence angle.
            
        Returns:
            total_loss: Total loss.
            loss_dict: Loss details.
        """
        if self.sar_simulator is None:
            return torch.tensor(0.0, device=sr_dem.device), {}
        
        # Expand scalar incidence angles to the same spatial size as the DEM.
        batch_size, _, height, width = hr_dem.shape
        incidence_angle_map = incidence_angle.view(batch_size, 1, 1, 1).expand(-1, -1, height, width)
            
        # Generate simulated SAR images with the SAR simulator.
        with torch.no_grad():
            # Generate simulated SAR from HR DEM as the reference.
            sar_hr = self.sar_simulator(hr_dem, incidence_angle_map)
            vv_sim_hr = sar_hr[:, 0:1]
            vh_sim_hr = sar_hr[:, 1:2]
            
            # Generate simulated SAR from SR DEM.
            sar_sr = self.sar_simulator(sr_dem, incidence_angle_map)
            vv_sim_sr = sar_sr[:, 0:1]
            vh_sim_sr = sar_sr[:, 1:2]
        
        # Compute L1 losses for VV and VH polarizations.
        vv_loss = self.l1_loss(vv_sim_sr, vv_sim_hr)
        vh_loss = self.l1_loss(vh_sim_sr, vh_sim_hr)
        
        # Average VV and VH losses.
        total_loss = (vv_loss + vh_loss) / 2
        
        # Return loss details.
        loss_dict = {
            'vv_loss': vv_loss.item(),
            'vh_loss': vh_loss.item()
        }
        
        return total_loss, loss_dict

# Slope loss.
class SlopeLoss(nn.Module):
    def __init__(self):
        super(SlopeLoss, self).__init__()
        # Initialize Sobel operators.
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.sobel_x.weight.data.copy_(sobel_x_kernel)
        self.sobel_y.weight.data.copy_(sobel_y_kernel)
        
        # Freeze Sobel operator parameters.
        for param in self.sobel_x.parameters():
            param.requires_grad = False
        for param in self.sobel_y.parameters():
            param.requires_grad = False
            
        self.l1_loss = nn.L1Loss()
    
    def compute_slope(self, x, dem_stats):
        """Compute slope in degrees."""
        # Ensure operators are on the input device.
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)
        
        # Compute gradients in the x and y directions in meters.
        batch_size = x.shape[0]
        slopes = []
        
        for i in range(batch_size):
            # Get current sample statistics for denormalization.
            mean = dem_stats['mean'][i].item()
            std = dem_stats['std'][i].item()
            
            # Compute real gradients in meters per pixel.
            grad_x = self.sobel_x(x[i:i+1]) * std
            grad_y = self.sobel_y(x[i:i+1]) * std
            
            # Compute slope in degrees.
            # slope = arctan(sqrt(dx^2 + dy^2))
            slope = torch.atan(torch.sqrt(grad_x.pow(2) + grad_y.pow(2)))
            slopes.append(slope)
        
        return torch.cat(slopes, dim=0)
    
    def forward(self, pred, target, dem_stats):
        """Compute slope loss.
        
        Args:
            pred: Predicted DEM.
            target: Ground-truth DEM.
            dem_stats: DEM statistics for denormalization.
            
        Returns:
            loss: Slope loss value.
            slope_metrics: Slope-related metrics.
        """
        # Compute slopes for predicted and ground-truth DEM data.
        pred_slope = self.compute_slope(pred, dem_stats)    # [B, 1, H, W]
        target_slope = self.compute_slope(target, dem_stats)  # [B, 1, H, W]
        
        # Compute the slope difference with L1 loss.
        slope_loss = self.l1_loss(pred_slope, target_slope)
        
        # Compute additional slope metrics.
        with torch.no_grad():
            # Convert to degrees.
            pred_slope_deg = torch.rad2deg(pred_slope)
            target_slope_deg = torch.rad2deg(target_slope)
            
            # Compute mean slope error in degrees.
            mean_slope_error = torch.mean(torch.abs(pred_slope_deg - target_slope_deg))
            
            # Compute maximum slope error in degrees.
            max_slope_error = torch.max(torch.abs(pred_slope_deg - target_slope_deg))
            
            # Compute slope RMSE in degrees.
            slope_rmse = torch.sqrt(torch.mean((pred_slope_deg - target_slope_deg) ** 2))
        
        slope_metrics = {
            'mean_slope_error': mean_slope_error.item(),  # Mean slope error in degrees.
            'max_slope_error': max_slope_error.item(),    # Maximum slope error in degrees.
            'slope_rmse': slope_rmse.item()              # Slope RMSE in degrees.
        }
        
        return slope_loss, slope_metrics


    
    def generate_simulated_sar(self, dem, incidence_angle):
        """Generate simulated SAR data."""
        simulated_sar = self.sar_simulator(dem, incidence_angle)
        vv_sim = simulated_sar[:, 0:1, :, :]
        vh_sim = simulated_sar[:, 1:2, :, :]
        return vv_sim, vh_sim
