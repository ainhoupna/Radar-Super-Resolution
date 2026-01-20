import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.san.san import SAN
from tqdm import tqdm

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "checkpoints_san/san_x2_20251219_193049_ssim_amp_aug/model_best.pth"
IMAGE_PATH = "/home/alumno/Desktop/datos/Computer Vision/Final Project/Zaragoza DANA pamplona/2023-09-01--08_10.png"
OUTPUT_PATH = "2023-09-03--08_10_SAN_SR.png"

SCALE_FACTOR = 2
PATCH_SIZE = 64 if SCALE_FACTOR == 2 else 32 # Match training: 64 for x2, 32 for x4
STRIDE = PATCH_SIZE // 2 # 50% overlap
N_FEATS = 64
N_RESGROUPS = 10
N_RESBLOCKS = 10
REDUCTION = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

class SANArgs:
    def __init__(self):
        self.n_resgroups = N_RESGROUPS
        self.n_resblocks = N_RESBLOCKS
        self.n_feats = N_FEATS
        self.reduction = REDUCTION
        self.scale = [SCALE_FACTOR]
        self.res_scale = 1.0
        self.rgb_range = 1.0
        self.n_colors = 1

def patch_origins(h, w, ps, stride):
    """Compute patch origins for a given image size, patch size and stride."""
    origins = []
    for i in range(0, h - ps + 1, stride):
        for j in range(0, w - ps + 1, stride):
            origins.append((i, j))
    
    # Ensure the edges are covered if they don't fit perfectly
    if (h - ps) % stride != 0:
        for j in range(0, w - ps + 1, stride):
            origins.append((h - ps, j))
    if (w - ps) % stride != 0:
        for i in range(0, h - ps + 1, stride):
            origins.append((i, w - ps))
    if (h - ps) % stride != 0 and (w - ps) % stride != 0:
        origins.append((h - ps, w - ps))
        
    return list(set(origins)) # Remove duplicates

def main():
    # 1. Load Model
    args = SANArgs()
    model = SAN(args).to(DEVICE)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Image
    img = Image.open(IMAGE_PATH).convert('L') # Single channel
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w = img_np.shape
    print(f"Original image size: {h}x{w}")

    # 3. Prepare for Inference
    ps = PATCH_SIZE
    out_ps = ps * SCALE_FACTOR
    out_h, out_w = h * SCALE_FACTOR, w * SCALE_FACTOR
    
    y_hat = torch.zeros((1, out_h, out_w), dtype=torch.float32)
    y_wei = torch.zeros((1, out_h, out_w), dtype=torch.float32)
    
    # Gradient filter for blending
    W = out_ps
    filterx = torch.concat([torch.arange(1, W//2+1, 1), torch.arange(W//2, 0, -1)]).unsqueeze(0).repeat(W, 1).float() / (W//2)
    filterx = filterx * filterx.t()
    filterx = filterx.to(torch.float32)

    coords = patch_origins(h, w, ps, STRIDE)
    
    transform = transforms.ToTensor()

    # 4. Inference loop
    print("Performing inference...")
    for (cx, cy) in tqdm(coords):
        # Crop LR patch
        xi_np = img_np[cx:cx+ps, cy:cy+ps]
        xi = transform(xi_np).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Forward
            with torch.amp.autocast('cuda'):
                sr = model(xi)
            sr = sr.clamp(0, 1).cpu().squeeze(0) # (1, out_ps, out_ps)

        # Output coordinates
        ocx, ocy = cx * SCALE_FACTOR, cy * SCALE_FACTOR
        
        # Accumulate with gradient blending
        y_hat[:, ocx:ocx+out_ps, ocy:ocy+out_ps] += sr * filterx
        y_wei[:, ocx:ocx+out_ps, ocy:ocy+out_ps] += filterx

    # 5. Normalize and Save
    print("Normalizing and saving...")
    y_hat /= (y_wei + 1e-8)
    y_hat = y_hat.clamp(0, 1).squeeze().numpy()
    
    res_img = Image.fromarray((y_hat * 255.0).astype(np.uint8))
    res_img.save(OUTPUT_PATH)
    print(f"Super-resolved image saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
