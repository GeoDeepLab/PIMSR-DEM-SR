import argparse
from pathlib import Path

class Config:
    """Training configuration."""
    def __init__(self):
        self.train_h5_path = r""
        self.train_csv_path = r""
        self.val_h5_path = r""
        self.val_csv_path = r""
        self.global_stats = r""
        self.simulator_weights = r''
        self.output_dir = r''
        self.batch_size = 16
        self.num_epochs = 200
        self.learning_rate = 1e-4
        self.num_workers = 4
        self.lr_step_size = 50
        self.lr_gamma = 0.5
        self.recon_weight = 1.0
        self.sar_weight = 0.1
        self.save_freq = 1
        self.vis_freq = 1
        self.resume = ""
        self.device = 'cuda'
        self.model_name = 'ResidualSARDEMGenerator'
        self.random_seed = 1008

    def update_from_args(self, args):
        self.train_h5_path = args.train_h5_path
        self.train_csv_path = args.train_csv_path
        self.val_h5_path = args.val_h5_path
        self.val_csv_path = args.val_csv_path
        self.global_stats = args.global_stats
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.output_dir = args.output_dir
        self.num_workers = args.num_workers
        self.lr_step_size = args.lr_step_size
        self.lr_gamma = args.lr_gamma
        self.recon_weight = args.recon_weight
        self.sar_weight = args.sar_weight
        self.save_freq = args.save_freq
        self.vis_freq = args.vis_freq
        if args.resume: self.resume = args.resume
        if args.simulator_weights: self.simulator_weights = args.simulator_weights
        if args.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        if hasattr(args, 'random_seed') and args.random_seed is not None:
            self.random_seed = args.random_seed

    def create_output_dirs(self):
        """Create output directories."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'checkpoints').mkdir(exist_ok=True)
        (output_dir / 'visualizations').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        return output_dir

    def get_paths_dict(self):
        return {'train_h5_path': self.train_h5_path, 'train_csv_path': self.train_csv_path,
                'val_h5_path': self.val_h5_path, 'val_csv_path': self.val_csv_path, 'global_stats': self.global_stats,
                'simulator_weights': self.simulator_weights, 'output_dir': self.output_dir}

    def get_training_params(self):
        return {'batch_size': self.batch_size, 'num_epochs': self.num_epochs, 'learning_rate': self.learning_rate,
                'num_workers': self.num_workers, 'lr_step_size': self.lr_step_size, 'lr_gamma': self.lr_gamma,
                'recon_weight': self.recon_weight, 'sar_weight': self.sar_weight}

    def print_config(self):
        print("=" * 60 + "\nTRAINING CONFIGURATION\n" + "=" * 60)
        print("\n[Data Paths]");
        [print(f"  {k}: {v}") for k, v in self.get_paths_dict().items()]
        print("\n[Training Parameters]");
        [print(f"  {k}: {v}") for k, v in self.get_training_params().items()]
        print(
            f"\n[Other Settings]\n  device: {self.device}\n  model_name: {self.model_name}\n  save_freq: {self.save_freq}\n  vis_freq: {self.vis_freq}\n  resume: {self.resume if self.resume else 'None'}\n  random_seed: {self.random_seed}")
        print("=" * 60)


def get_config_from_args():
    """Build the configuration from command-line arguments."""
    parser = argparse.ArgumentParser(description='SAR Enhanced DEM Super Resolution Training')
    parser.add_argument('--train_h5_path', type=str,
                        default=r"path/to/your/Train_data/train_data_real_lr.h5",
                        help='Path to the training HDF5 file with real LR data')
    parser.add_argument('--train_csv_path', type=str,
                        default=r"path/to/your/Train_data/data_index.csv",
                        help='Path to the training index CSV file')
    parser.add_argument('--val_h5_path', type=str,
                        default=r"path/to/your/Val_data/val_data_real_lr.h5",
                        help='Path to the validation HDF5 file with real LR data')
    parser.add_argument('--val_csv_path', type=str,
                        default=r"path/to/your/Val_data/data_index.csv",
                        help='Path to the validation index CSV file')
    parser.add_argument('--global_stats', type=str,
                        default=r'path/to/your/Train_data/global_stats.json',
                        help='Global statistics JSON file')
    parser.add_argument('--simulator_weights', type=str,
                        default=r'path/to/your/checkpoints/best_precise_simulator.pth',
                        help='SAR simulator pretrained weights')
    parser.add_argument('--output_dir', type=str, default=r'path/to/your/results/train',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--lr_step_size', type=int, default=25, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate gamma')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--sar_weight', type=float, default=1, help='SAR loss weight')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency')
    parser.add_argument('--vis_freq', type=int, default=1, help='Visualization frequency')
    parser.add_argument('--random_seed', type=int, default=1008, help='Random seed for reproducibility')

    args = parser.parse_args()
    config = Config()
    config.update_from_args(args)
    return config
