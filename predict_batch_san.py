import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.san.san import SAN
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "checkpoints_san/san_x2_20251219_193049_ssim_amp_aug/model_best.pth"
INPUT_DIR = "/home/alumno/Desktop/datos/Computer Vision/Final Project/Zaragoza DANA pamplona/"
OUTPUT_DIR = "predictions_dana_zaragoza_32"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE_FACTOR = 2
PATCH_SIZE = 64 if SCALE_FACTOR == 2 else 32 # Match training: 64 for x2, 32 for x4
STRIDE = PATCH_SIZE // 2 
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
    origins = []
    for i in range(0, h - ps + 1, stride):
        for j in range(0, w - ps + 1, stride):
            origins.append((i, j))
    if (h - ps) % stride != 0:
        for j in range(0, w - ps + 1, stride):
            origins.append((h - ps, j))
    if (w - ps) % stride != 0:
        for i in range(0, h - ps + 1, stride):
            origins.append((i, w - ps))
    if (h - ps) % stride != 0 and (w - ps) % stride != 0:
        origins.append((h - ps, w - ps))
    return list(set(origins))

def apply_radar_colormap(img_np):
    """Applies a jet colormap and sets zero values to black."""
    colormap = cm.get_cmap('jet')
    colored_img = colormap(img_np)
    # Mask for zero values (no rain)
    mask = img_np < 0.01 # Threshold for zero
    colored_img[mask, :3] = 0
    return (colored_img[:, :, :3] * 255).astype(np.uint8)

def main():
    # 1. Load Model
    args = SANArgs()
    model = SAN(args).to(DEVICE)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Get Image List
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    files.sort()
    print(f"Found {len(files)} images to process.")

    # 3. Prepare Blending Filter
    ps = PATCH_SIZE
    out_ps = ps * SCALE_FACTOR
    W = out_ps
    filterx = torch.concat([torch.arange(1, W//2+1, 1), torch.arange(W//2, 0, -1)]).unsqueeze(0).repeat(W, 1).float() / (W//2)
    filterx = (filterx * filterx.t()).to(torch.float32)
    
    transform = transforms.ToTensor()

    # 4. Process Images
    for filename in tqdm(files, desc="Processing Images"):
        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape
        
        out_h, out_w = h * SCALE_FACTOR, w * SCALE_FACTOR
        y_hat = torch.zeros((1, out_h, out_w), dtype=torch.float32)
        y_wei = torch.zeros((1, out_h, out_w), dtype=torch.float32)
        
        coords = patch_origins(h, w, ps, STRIDE)
        
        # SAN Inference
        for (cx, cy) in coords:
            xi_np = img_np[cx:cx+ps, cy:cy+ps]
            xi = transform(xi_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    sr = model(xi)
                sr = sr.clamp(0, 1).cpu().squeeze(0)
            ocx, ocy = cx * SCALE_FACTOR, cy * SCALE_FACTOR
            y_hat[:, ocx:ocx+out_ps, ocy:ocy+out_ps] += sr * filterx
            y_wei[:, ocx:ocx+out_ps, ocy:ocy+out_ps] += filterx

        y_hat /= (y_wei + 1e-8)
        san_sr = y_hat.clamp(0, 1).squeeze().numpy()
        
        # Bicubic Upsampling
        bicubic_sr = np.array(img.resize((out_w, out_h), Image.BICUBIC)).astype(np.float32) / 255.0
        
        # Original Upscaled (for comparison)
        original_up = np.array(img.resize((out_w, out_h), Image.NEAREST)).astype(np.float32) / 255.0

        # Apply Colormaps
        orig_color = apply_radar_colormap(original_up)
        bicubic_color = apply_radar_colormap(bicubic_sr)
        san_color = apply_radar_colormap(san_sr)
        
        # Create Comparison Image (Horizontal Concatenation)
        # Add labels text
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_color)
        axes[0].set_title("Original (Upscaled)")
        axes[0].axis('off')
        
        axes[1].imshow(bicubic_color)
        axes[1].set_title("Bicubic x2")
        axes[1].axis('off')
        
        axes[2].imshow(san_color)
        axes[2].set_title("SAN x2")
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(OUTPUT_DIR, f"comparison_{filename}")
        plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Save individual colored results
        Image.fromarray(san_color).save(os.path.join(OUTPUT_DIR, f"san_{filename}"))
        Image.fromarray(bicubic_color).save(os.path.join(OUTPUT_DIR, f"bicubic_{filename}"))

    print(f"Batch processing complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
