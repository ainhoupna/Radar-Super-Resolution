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
import pyproj
import rasterio
from rasterio.transform import from_origin
import contextily as ctx

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "checkpoints_san/san_x2_20251219_193049_ssim_amp_aug/model_best.pth"
INPUT_DIR = "/home/alumno/Desktop/datos/Computer Vision/Final Project/Zaragoza DANA pamplona/"
OUTPUT_DIR = "georef_dana_results_x2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time range
START_TIME = "2023-09-02--07_00"
END_TIME = "2023-09-02--16_00"

# Model Params
SCALE_FACTOR = 2       
PATCH_SIZE = 64 if SCALE_FACTOR == 2 else 32 # Match training: 64 for x2, 32 for x4
STRIDE = PATCH_SIZE // 2 
N_FEATS = 64
N_RESGROUPS = 10
N_RESBLOCKS = 10
REDUCTION = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Georef Params
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
    """Calculate the extent in LCC and Web Mercator."""
    proj_lcc = pyproj.Proj("+proj=lcc +lat_1=33.5 +lat_2=46.5 +lat_0=40 +lon_0=0 +ellps=WGS84 +units=m +no_defs")
    x_radar, y_radar = proj_lcc(lon, lat)
    
    x_min = x_radar - (width // 2) * res
    x_max = x_radar + (width // 2) * res
    y_min = y_radar - (height // 2) * res
    y_max = y_radar + (height // 2) * res
    
    # Return extent for matplotlib [xmin, xmax, ymin, ymax]
    return [x_min, x_max, y_min, y_max], proj_lcc.srs

def main():
    # 1. Load Model
    args = SANArgs()
    model = SAN(args).to(DEVICE)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Filter Files
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    all_files.sort()
    files = [f for f in all_files if START_TIME <= f.replace(".png", "") <= END_TIME]
    print(f"Processing {len(files)} images in range {START_TIME} to {END_TIME}.")

    # 3. Prepare Blending Filter
    ps = PATCH_SIZE
    out_ps = ps * SCALE_FACTOR
    W = out_ps
    filterx = torch.concat([torch.arange(1, W//2+1, 1), torch.arange(W//2, 0, -1)]).unsqueeze(0).repeat(W, 1).float() / (W//2)
    filterx = (filterx * filterx.t()).to(torch.float32)
    
    transform = transforms.ToTensor()
    
    # Georef Extent
    extent_lcc, lcc_crs_str = get_extent(LON_RADAR, LAT_RADAR, WIDTH_ORIG, HEIGHT_ORIG, RES_METROS_ORIG)
    lcc_crs = pyproj.CRS.from_user_input(lcc_crs_str)

    # 4. Process Images
    for filename in tqdm(files, desc="Generating Georef TIFs"):
        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape
        
        # SAN Inference
        out_h, out_w = h * SCALE_FACTOR, w * SCALE_FACTOR
        y_hat = torch.zeros((1, out_h, out_w), dtype=torch.float32)
        y_wei = torch.zeros((1, out_h, out_w), dtype=torch.float32)
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
        
        # 5. Warp to EPSG:3857 for reliable plotting with OSM
        def warp_to_3857(data, extent_lcc, lcc_crs_str):
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            from rasterio.transform import from_bounds
            
            src_crs = lcc_crs_str
            dst_crs = 'EPSG:3857'
            
            # Source transform: from_bounds(west, south, east, north, width, height)
            src_transform = from_bounds(extent_lcc[0], extent_lcc[2], extent_lcc[1], extent_lcc[3], data.shape[1], data.shape[0])
            
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], *extent_lcc
            )
            
            dst_data = np.zeros((dst_height, dst_width), dtype=np.float32)
            
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            
            # extent = [left, right, bottom, top]
            extent_3857 = [dst_transform[2], dst_transform[2] + dst_width * dst_transform[0],
                           dst_transform[5] + dst_height * dst_transform[4], dst_transform[5]]
            
            return dst_data, extent_3857

        img_3857, extent_3857 = warp_to_3857(img_np, extent_lcc, lcc_crs_str)
        san_3857, _ = warp_to_3857(san_sr, extent_lcc, lcc_crs_str)

        # 6. Visualization with OSM
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Colormap: Jet with transparency for zero
        cmap = plt.get_cmap('jet').copy()
        cmap.set_under(alpha=0) 
        
        # Original
        axes[0].imshow(img_3857, extent=extent_3857, origin='upper', cmap=cmap, vmin=0.01, alpha=0.8, zorder=10)
        axes[0].set_title(f"Original (1km) - {filename}")
        try:
            ctx.add_basemap(axes[0], source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
        except Exception as e:
            print(f"Error adding basemap to axes[0]: {e}")
        axes[0].set_xlim(extent_3857[0], extent_3857[1])
        axes[0].set_ylim(extent_3857[2], extent_3857[3])
        axes[0].axis('off')
        
        # SAN
        axes[1].imshow(san_3857, extent=extent_3857, origin='upper', cmap=cmap, vmin=0.01, alpha=0.8, zorder=10)
        axes[1].set_title(f"SAN Super-Resolution (250m) - {filename}")
        try:
            ctx.add_basemap(axes[1], source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
        except Exception as e:
            print(f"Error adding basemap to axes[1]: {e}")
        axes[1].set_xlim(extent_3857[0], extent_3857[1])
        axes[1].set_ylim(extent_3857[2], extent_3857[3])
        axes[1].axis('off')
        
        plt.tight_layout()
        output_name = filename.replace(".png", ".tif")
        plt.savefig(os.path.join(OUTPUT_DIR, output_name), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Processing complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
