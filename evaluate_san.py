import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr_skimage
from skimage.metrics import structural_similarity as calculate_ssim_skimage
from models.san.san import SAN

# ================= CONFIGURATION =================
TEST_DIR = "dataset_split/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    2: "checkpoints_san/san_x2_20251219_193049_ssim_amp_aug/model_best.pth",
    4: "checkpoints_san/san_x4_20251222_192504_ssim_amp_aug/model_best.pth"
}

# SAN Params (Must match training)
N_RESGROUPS = 10
N_RESBLOCKS = 10
N_FEATS = 64
REDUCTION = 16
# =================================================

class SANArgs:
    def __init__(self, scale):
        self.n_resgroups = N_RESGROUPS
        self.n_resblocks = N_RESBLOCKS
        self.n_feats = N_FEATS
        self.reduction = REDUCTION
        self.scale = [scale]
        self.res_scale = 1.0
        self.rgb_range = 1.0
        self.n_colors = 1

def compute_metrics(target_hr, image_sr):
    """Calculates PSNR and SSIM between the two images."""
    target_hr_np = (target_hr.squeeze().cpu().numpy() * 255.0).astype(np.float32)
    image_sr_np = (image_sr.squeeze().cpu().numpy() * 255.0).astype(np.float32)
    
    psnr = calculate_psnr_skimage(target_hr_np, image_sr_np, data_range=255)
    ssim = calculate_ssim_skimage(target_hr_np, image_sr_np, data_range=255, channel_axis=None) 
    return psnr, ssim

def evaluate(scale):
    checkpoint_path = CHECKPOINTS.get(scale)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint for x{scale} not found at {checkpoint_path}")
        return None

    print(f"\n--- Evaluating SAN x{scale} ---")
    args = SANArgs(scale)
    model = SAN(args).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.safetensors')]
    lr_key = f"LR_{scale}x"
    
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for filename in tqdm(test_files, desc=f"x{scale} Evaluation"):
            path = os.path.join(TEST_DIR, filename)
            tensors = load_file(path)
            
            lr = tensors[lr_key].to(DEVICE) # (1, H, W)
            hr = tensors['HR'].to(DEVICE)   # (1, H, W)
            
            # Add batch dimension
            lr = lr.unsqueeze(0)
            
            with torch.amp.autocast('cuda'):
                sr = model(lr)
            
            sr = torch.clamp(sr, 0, 1).squeeze(0) # (1, H, W)
            
            psnr, ssim = compute_metrics(hr, sr)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"Results for SAN x{scale}:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    results = {}
    for scale in [2, 4]:
        res = evaluate(scale)
        if res:
            results[scale] = res
    
    print("\n" + "="*30)
    print("FINAL SAN TEST METRICS")
    print("="*30)
    for scale, (psnr, ssim) in results.items():
        print(f"x{scale}: PSNR = {psnr:.4f} dB, SSIM = {ssim:.4f}")
    print("="*30)
