import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyproj
import contextily as ctx
from models.san.san import SAN
import torchvision.transforms as transforms

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "checkpoints_san/san_x2_20251219_193049_ssim_amp_aug/model_best.pth"
IMAGE_PATH = "/home/alumno/Desktop/datos/Computer Vision/Final Project/Zaragoza DANA pamplona/2023-09-02--08_10.png"
OUTPUT_DEBUG = "debug_georef_0810.png"

SCALE_FACTOR = 2
PATCH_SIZE = 48        
STRIDE = PATCH_SIZE // 2 
N_FEATS = 64
N_RESGROUPS = 10
N_RESBLOCKS = 10
REDUCTION = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LON_RADAR, LAT_RADAR = -0.5458, 41.7339
RES_METROS_ORIG = 1000
WIDTH_ORIG, HEIGHT_ORIG = 480, 480
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

def get_extent(lon, lat, width, height, res):
    proj_lcc = pyproj.Proj("+proj=lcc +lat_1=33.5 +lat_2=46.5 +lat_0=40 +lon_0=0 +ellps=WGS84 +units=m +no_defs")
    x_radar, y_radar = proj_lcc(lon, lat)
    x_min = x_radar - (width // 2) * res
    x_max = x_radar + (width // 2) * res
    y_min = y_radar - (height // 2) * res
    y_max = y_radar + (height // 2) * res
    return [x_min, x_max, y_min, y_max], proj_lcc.srs

def main():
    # 1. Load Model
    args = SANArgs()
    model = SAN(args).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Image
    img = Image.open(IMAGE_PATH).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    print(f"Original image stats: min={img_np.min()}, max={img_np.max()}, mean={img_np.mean()}")
    
    # 3. SAN Inference
    h, w = img_np.shape
    out_h, out_w = h * SCALE_FACTOR, w * SCALE_FACTOR
    y_hat = torch.zeros((1, out_h, out_w), dtype=torch.float32)
    y_wei = torch.zeros((1, out_h, out_w), dtype=torch.float32)
    
    ps = PATCH_SIZE
    out_ps = ps * SCALE_FACTOR
    W = out_ps
    filterx = torch.concat([torch.arange(1, W//2+1, 1), torch.arange(W//2, 0, -1)]).unsqueeze(0).repeat(W, 1).float() / (W//2)
    filterx = (filterx * filterx.t()).to(torch.float32)
    
    transform = transforms.ToTensor()
    coords = patch_origins(h, w, ps, STRIDE)
    
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
    print(f"SAN SR stats: min={san_sr.min()}, max={san_sr.max()}, mean={san_sr.mean()}")

    # 4. Plotting Debug
    extent_lcc, lcc_crs = get_extent(LON_RADAR, LAT_RADAR, WIDTH_ORIG, HEIGHT_ORIG, RES_METROS_ORIG)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    cmap = cm.jet
    cmap.set_under(alpha=0)
    
    # Row 1: Without Basemap
    axes[0, 0].imshow(img_np, extent=extent_lcc, origin='upper', cmap=cmap, vmin=0.01)
    axes[0, 0].set_title("Original (No Basemap)")
    
    axes[0, 1].imshow(san_sr, extent=extent_lcc, origin='upper', cmap=cmap, vmin=0.01)
    axes[0, 1].set_title("SAN SR (No Basemap)")
    
    # Row 2: With Basemap
    axes[1, 0].imshow(img_np, extent=extent_lcc, origin='upper', cmap=cmap, vmin=0.01, alpha=0.8, zorder=10)
    ctx.add_basemap(axes[1, 0], crs=lcc_crs, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
    axes[1, 0].set_title("Original (With Basemap)")
    
    axes[1, 1].imshow(san_sr, extent=extent_lcc, origin='upper', cmap=cmap, vmin=0.01, alpha=0.8, zorder=10)
    ctx.add_basemap(axes[1, 1], crs=lcc_crs, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
    axes[1, 1].set_title("SAN SR (With Basemap)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DEBUG)
    print(f"Debug plot saved to {OUTPUT_DEBUG}")

if __name__ == "__main__":
    main()
