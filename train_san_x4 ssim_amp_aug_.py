import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from piq import ssim as ssim_loss  # Differentiable SSIM for training

# Import SAN model
from models.san.san import SAN

# ================= CONFIGURATION =================
TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

SCALE_FACTOR = 4       
N_RESGROUPS = 10       # SAN specific: number of residual groups
N_RESBLOCKS = 10       # SAN specific: number of residual blocks per group
N_FEATS = 64           
REDUCTION = 16         # SAN specific: reduction factor for attention
RES_SCALE = 1.0        

BATCH_SIZE = 32        
LEARNING_RATE = 1e-4   
EPOCHS = 60           
PATIENCE = 7          
ALPHA_SSIM = 0.2      # Weight for SSIM loss (Total Loss = L1 + ALPHA * (1-SSIM))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_ROOT_DIR = "checkpoints_san"
MLFLOW_EXPERIMENT_NAME = "AEMET_Radar_SAN"
# =================================================

# --- 0. Args Class for SAN ---
class SANArgs:
    def __init__(self):
        self.n_resgroups = N_RESGROUPS
        self.n_resblocks = N_RESBLOCKS
        self.n_feats = N_FEATS
        self.reduction = REDUCTION
        self.scale = [SCALE_FACTOR]
        self.res_scale = RES_SCALE
        self.rgb_range = 1.0
        self.n_colors = 1

# --- 1. Early Stopping Class ---
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_psnr):
        score = val_psnr
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# --- 2. Dataset Loader ---
class RadarDataset(Dataset):
    def __init__(self, root_dir, scale, augment=False):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.safetensors')]
        self.lr_key = f"LR_{scale}x"
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        tensors = load_file(path)
        lr = tensors[self.lr_key]
        hr = tensors['HR']

        if self.augment:
            lr, hr = self._dihedral_augmentation(lr, hr)
        return lr, hr

    def _dihedral_augmentation(self, lr, hr):
        rot_k = random.randint(0, 3)
        lr = torch.rot90(lr, k=rot_k, dims=[1, 2])
        hr = torch.rot90(hr, k=rot_k, dims=[1, 2])
        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr

# --- 3. Utilities ---
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else torch.tensor(100.0)

# --- 4. Main Training Loop ---
def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"san_x{SCALE_FACTOR}_{timestamp}_ssim_amp_aug"
    CURRENT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT_DIR, RUN_NAME)
    os.makedirs(CURRENT_CHECKPOINT_DIR, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "model": "SAN",
            "scale": SCALE_FACTOR,
            "res_groups": N_RESGROUPS,
            "res_blocks": N_RESBLOCKS,
            "features": N_FEATS,
            "reduction": REDUCTION,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "alpha_ssim": ALPHA_SSIM,
            "mixed_precision": True,
            "augmentation": "none"
        })

        train_loader = DataLoader(RadarDataset(TRAIN_DIR, SCALE_FACTOR, augment=False), 
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(RadarDataset(VAL_DIR, SCALE_FACTOR, augment=False), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        args = SANArgs()
        model = SAN(args).to(DEVICE)
        
        criterion_l1 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=PATIENCE, delta=0.01)

        scaler = torch.amp.GradScaler('cuda')
        best_psnr = 0.0

        for epoch in range(EPOCHS):
            model.train()
            train_loss_accum = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for lr, hr in loop:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                optimizer.zero_grad()
                # Forward with AMP
                with torch.amp.autocast('cuda'):
                    sr = model(lr)
                    l1 = criterion_l1(sr, hr)
                    ssim_val = ssim_loss(torch.clamp(sr, 0, 1), hr, data_range=1.0)
                    loss = l1 + ALPHA_SSIM * (1.0 - ssim_val)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_accum += loss.item()
                loop.set_postfix(loss=loss.item(), ssim=ssim_val.item())

            model.eval()
            val_psnr, val_ssim = 0, 0
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        sr = torch.clamp(model(lr), 0, 1)
                    val_psnr += calculate_psnr(sr.float(), hr).item()
                    val_ssim += ssim_loss(sr.float(), hr, data_range=1.0).item()
            
            avg_psnr = val_psnr / len(val_loader)
            avg_ssim = val_ssim / len(val_loader)

            mlflow.log_metrics({
                "train_loss": train_loss_accum / len(train_loader),
                "val_psnr": avg_psnr,
                "val_ssim": avg_ssim,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)

            print(f"Epoch {epoch+1}: PSNR {avg_psnr:.2f} dB | SSIM {avg_ssim:.4f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), os.path.join(CURRENT_CHECKPOINT_DIR, "model_best.pth"))
                mlflow.pytorch.log_model(model, "model_best")
                print(f"--> New Best Model Saved! ({best_psnr:.2f} dB)")

            scheduler.step()
            early_stopping(avg_psnr)
            if early_stopping.early_stop:
                print("[INFO] Early stopping triggered.")
                break

    print("Training Complete.")

if __name__ == "__main__":
    main()
