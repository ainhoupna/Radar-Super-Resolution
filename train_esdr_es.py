#!/usr/bin/env python
# coding: utf-8

import os
import math
import time # Added for unique timestamp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

# ================= CONFIGURATION =================
# Dataset paths
TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

# Model Parameters
SCALE_FACTOR = 4       # Change to 4 when ready
N_RESBLOCKS = 32       
N_FEATS = 64           
RES_SCALE = 0.1        

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4   
EPOCHS = 150           
PATIENCE = 7          # Epochs to wait for improvement before stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_ROOT_DIR = "checkpoints_edsr" # Root folder for all checkpoints
# =================================================

# --- 0. Early Stopping Class ---
class EarlyStopping:
    """Stops training if validation metric doesn't improve after a given patience."""
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
            # No significant improvement
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found
            self.best_score = score
            self.counter = 0

# --- 1. EDSR Model Architecture (Same as before) ---
class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        scale = args['scale']
        kernel_size = 3 
        act = nn.ReLU(True)

        m_head = [nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2)]
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=RES_SCALE)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        m_tail = [
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size//2),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, kernel_size, padding=kernel_size//2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# --- 2. Dataset Loader ---
class RadarDataset(Dataset):
    def __init__(self, root_dir, scale):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.safetensors')]
        self.scale = scale
        self.lr_key = f"LR_{scale}x"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        tensors = load_file(path)
        hr = tensors['HR']
        lr = tensors[self.lr_key]
        return lr, hr

# --- 3. Utilities ---
def calculate_psnr(img1, img2):
    # Image range is [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# --- 4. Main Training Loop ---
def main():
    # --- Checkpoint Setup with Timestamp ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"edsr_x{SCALE_FACTOR}_{timestamp}"
    CURRENT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT_DIR, RUN_NAME)
    
    os.makedirs(CURRENT_CHECKPOINT_DIR, exist_ok=True)
    print(f"--- Starting Training Run: {RUN_NAME} on {DEVICE} ---")
    print(f"Checkpoints will be saved in: {CURRENT_CHECKPOINT_DIR}")

    # Setup Data
    train_dataset = RadarDataset(TRAIN_DIR, scale=SCALE_FACTOR)
    val_dataset = RadarDataset(VAL_DIR, scale=SCALE_FACTOR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Setup Model
    args = {'n_resblocks': N_RESBLOCKS, 'n_feats': N_FEATS, 'scale': SCALE_FACTOR}
    model = EDSR(args).to(DEVICE)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, delta=0) # delta=0 dB minimal improvement
    best_psnr = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for lr, hr in loop:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_psnr = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = model(lr)
                sr = torch.clamp(sr, 0, 1)
                val_psnr += calculate_psnr(sr, hr).item()
        
        avg_psnr = val_psnr / len(val_loader)
        
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f} | Val PSNR: {avg_psnr:.2f} dB")
        
        # Save Best Model Logic
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            # Save using the unique run folder
            save_path = os.path.join(CURRENT_CHECKPOINT_DIR, f"model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Model Saved! ({best_psnr:.2f} dB)")
        
        scheduler.step()

        # Check Early Stopping
        early_stopping(avg_psnr)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered. Training stopped at epoch {epoch+1}.")
            print(f"Best PSNR achieved: {best_psnr:.2f} dB")
            break

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()