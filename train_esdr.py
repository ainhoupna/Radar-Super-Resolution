#!/usr/bin/env python
# coding: utf-8

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from tqdm import tqdm

# ================= CONFIGURATION =================
# Dataset paths (Created by the split script)
TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

# Model Parameters
SCALE_FACTOR = 4       # Change to 4 when ready
N_RESBLOCKS = 32       # Depth of the network (16 or 32 are standard)
N_FEATS = 64           # Width of the network (64 is standard)
RES_SCALE = 0.1        # Residual scaling factor (stabilizes training)

# Training Hyperparameters
BATCH_SIZE = 16        # Adjust based on your GPU VRAM (Try 32 if 16 is easy)
LEARNING_RATE = 1e-4   # Standard EDSR learning rate
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints_edsr"
# =================================================


# --- 1. EDSR Model Architecture ---

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

        # RGB mean for normalization (Optional for radar, skipping here as data is 0-1)
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Head
        m_head = [nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2)]

        # Body
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=RES_SCALE)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))

        # Tail (Upsampler)
        m_tail = [
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size//2),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, kernel_size, padding=kernel_size//2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x) # Skipping mean shift for radar data
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        # x = self.add_mean(x)
        return x

# --- 2. Dataset Loader ---

class RadarDataset(Dataset):
    def __init__(self, root_dir, scale):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.safetensors')]
        self.scale = scale
        # Determine the key to look for in safetensors
        self.lr_key = f"LR_{scale}x"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        tensors = load_file(path)
        
        # Load HR and specific LR (e.g., LR_2x)
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"--- Training EDSR x{SCALE_FACTOR} on {DEVICE} ---")

    # 1. Setup Data
    train_dataset = RadarDataset(TRAIN_DIR, scale=SCALE_FACTOR)
    val_dataset = RadarDataset(VAL_DIR, scale=SCALE_FACTOR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Setup Model
    args = {'n_resblocks': N_RESBLOCKS, 'n_feats': N_FEATS, 'scale': SCALE_FACTOR}
    model = EDSR(args).to(DEVICE)
    
    # 3. Optimization
    criterion = nn.L1Loss() # L1 Loss is preferred for SR
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    # Simple scheduler to decay LR every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_psnr = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # --- Training ---
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
        
        # --- Validation ---
        model.eval()
        val_psnr = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = model(lr)
                # Clamp outputs to valid range [0, 1] before metric
                sr = torch.clamp(sr, 0, 1)
                val_psnr += calculate_psnr(sr, hr).item()
        
        avg_psnr = val_psnr / len(val_loader)
        
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f} | Val PSNR: {avg_psnr:.2f} dB")
        
        # Save Best Model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_path = os.path.join(CHECKPOINT_DIR, f"edsr_x{SCALE_FACTOR}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Model Saved! ({best_psnr:.2f} dB)")
        
        scheduler.step()

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()