import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr_skimage
from skimage.metrics import structural_similarity as calculate_ssim_skimage

# =================================================================
# --- 0. MODEL ARCHITECTURE (MUST MATCH TRAINING SCRIPT EXACTLY) ---
# =================================================================

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):
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
            ResBlock(n_feats, kernel_size, act=act, res_scale=0.1)
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

# ==========================================
# --- CONFIGURATION ---
# ==========================================
TEST_DIR = "dataset_split/test" 

# Checkpoint paths (UPDATE THESE)
# CHECKPOINT_PATH_X2 = "checkpoints_edsr/edsr_x2_20251215_164500/model_best.pth" 
# CHECKPOINT_PATH_X4 = "checkpoints_edsr/edsr_x4_20251216_100000/model_best.pth" 



# Full path to the CHECKPOINT for the x2 model
CHECKPOINT_PATH_X2 = "checkpoints_edsr/edsr_x2_20251215_180313/model_best.pth" 

# Full path to the CHECKPOINT for the x4 model
# NOTE: Replace this with the actual path once you train the x4 model.
CHECKPOINT_PATH_X4 = "checkpoints_edsr/edsr_x4_20251216_085313/model_best.pth" 

MODEL_PARAMS = {
    'n_resblocks': 16,
    'n_feats': 64,
}

NUM_VISUAL_SAMPLES = 5
OUTPUT_ROOT_DIR = "evaluation_comparison_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


# --- FIXED UTILITY FUNCTIONS ---

def load_model(checkpoint_path, scale_factor, device):
    """Loads the model architecture and its trained weights for a specific scale."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping model load.")
        return None

    params = MODEL_PARAMS.copy()
    params['scale'] = scale_factor
    model = EDSR(params).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model x{scale_factor} loaded successfully.")
    return model

# --- FIXED: upsample_bicubic function ---
def upsample_bicubic(lr_tensor, scale_factor):
    """
    Upsamples the LR tensor using standard Bicubic interpolation.
    Fixes the 3D/4D input error by handling device and batch dimension explicitly.
    Assumes lr_tensor input is (C, H, W).
    """
    # 1. Ensure the tensor is on the correct device (GPU/CPU)
    lr_tensor = lr_tensor.to(DEVICE) 

    # 2. Add batch dimension (N=1) for F.interpolate (N, C, H, W)
    if lr_tensor.dim() == 3:
        lr_input = lr_tensor.unsqueeze(0)
    elif lr_tensor.dim() == 4:
        lr_input = lr_tensor
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, got {lr_tensor.dim()}D")
    
    # 3. Bicubic Interpolation
    bicubic_sr = F.interpolate(
        lr_input, 
        scale_factor=scale_factor, 
        mode='bicubic', 
        align_corners=False
    )
    
    # 4. Remove batch dimension and move back to CPU for metric calculation/plotting
    bicubic_sr = bicubic_sr.squeeze(0).cpu()
    
    return torch.clamp(bicubic_sr, 0, 1)

def compute_metrics(target_hr, image_sr):
    """Calculates PSNR and SSIM between the two images."""
    # Convert Tensors (C, H, W) to NumPy (H, W) and scale to 0-255 range
    target_hr_np = (target_hr.squeeze().cpu().numpy() * 255.0).astype(np.float32)
    image_sr_np = (image_sr.squeeze().cpu().numpy() * 255.0).astype(np.float32)
    
    # PSNR
    psnr = calculate_psnr_skimage(target_hr_np, image_sr_np, data_range=255)
    
    # SSIM
    ssim = calculate_ssim_skimage(target_hr_np, image_sr_np, data_range=255, channel_axis=None) 
    
    return psnr, ssim

# --- MAIN EVALUATION FUNCTIONS ---

def calculate_metrics(model, scale_factor):
    """Performs metric calculation over the entire test set."""
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.safetensors')]
    if not test_files:
        print(f"Error: No test files found in {TEST_DIR}.")
        return None

    print(f"\n--- Calculating Metrics for x{scale_factor} (Total patches: {len(test_files)}) ---")
    
    lr_key = f"LR_{scale_factor}x"
    
    edsr_psnr_list, edsr_ssim_list = [], []
    bicubic_psnr_list, bicubic_ssim_list = [], []

    with torch.no_grad():
        for filename in tqdm(test_files, desc=f"x{scale_factor} Metrics"):
            path = os.path.join(TEST_DIR, filename)
            tensors = load_file(path)
            
            # Load LR/HR (C, H, W)
            lr_tensor_cpu = tensors[lr_key].squeeze(0).unsqueeze(0) # (C, H, W)
            hr_tensor = tensors['HR'].squeeze(0).unsqueeze(0) # (C, H, W)
            
            # 1. EDSR Prediction
            if model is not None:
                # Needs (N, C, H, W) input on the device
                lr_input_edsr = lr_tensor_cpu.unsqueeze(0).to(DEVICE)
                edsr_sr = model(lr_input_edsr).squeeze(0).cpu() # Output back to CPU
                edsr_sr = torch.clamp(edsr_sr, 0, 1)
                psnr_edsr, ssim_edsr = compute_metrics(hr_tensor, edsr_sr)
                edsr_psnr_list.append(psnr_edsr)
                edsr_ssim_list.append(ssim_edsr)

            # 2. Bicubic Baseline (Uses (C, H, W) tensor, function handles device move)
            bicubic_sr = upsample_bicubic(lr_tensor_cpu, scale_factor)

            # 3. Compute Metrics (All tensors are now on CPU)
            psnr_bicubic, ssim_bicubic = compute_metrics(hr_tensor, bicubic_sr)
            
            bicubic_psnr_list.append(psnr_bicubic)
            bicubic_ssim_list.append(ssim_bicubic)

    avg_edsr_psnr = np.mean(edsr_psnr_list) if edsr_psnr_list else 0.0
    avg_edsr_ssim = np.mean(edsr_ssim_list) if edsr_ssim_list else 0.0
    avg_bicubic_psnr = np.mean(bicubic_psnr_list)
    avg_bicubic_ssim = np.mean(bicubic_ssim_list)

    results = {
        'scale': scale_factor,
        'EDSR_PSNR': avg_edsr_psnr,
        'EDSR_SSIM': avg_edsr_ssim,
        'BICUBIC_PSNR': avg_bicubic_psnr,
        'BICUBIC_SSIM': avg_bicubic_ssim
    }
    return results

    avg_edsr_psnr = np.mean(edsr_psnr_list)
    avg_edsr_ssim = np.mean(edsr_ssim_list)
    avg_bicubic_psnr = np.mean(bicubic_psnr_list)
    avg_bicubic_ssim = np.mean(bicubic_ssim_list)

    results = {
        'scale': scale_factor,
        'EDSR_PSNR': avg_edsr_psnr,
        'EDSR_SSIM': avg_edsr_ssim,
        'BICUBIC_PSNR': avg_bicubic_psnr,
        'BICUBIC_SSIM': avg_bicubic_ssim
    }
    return results


def visualize_comparison(model, scale_factor, num_samples):
    """Generates comparison plots for random samples."""
    if model is None:
        print(f"Skipping visualization for x{scale_factor} due to missing model.")
        return

    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.safetensors')]
    if len(test_files) < num_samples:
        samples = test_files
        if not samples: return
    else:
        samples = random.sample(test_files, num_samples)

    output_dir = os.path.join(OUTPUT_ROOT_DIR, f"visual_x{scale_factor}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Generating {len(samples)} Visualizations for x{scale_factor} ---")
    
    lr_key = f"LR_{scale_factor}x"
    
    with torch.no_grad():
        for filename in tqdm(samples, desc=f"x{scale_factor} Visuals"):
            path = os.path.join(TEST_DIR, filename)
            tensors = load_file(path)
            
            # Load LR/HR (C, H, W)
            lr_tensor_cpu = tensors[lr_key].squeeze(0).unsqueeze(0) # (C, H, W)
            hr_tensor = tensors['HR'].squeeze(0).unsqueeze(0)
            
            lr_input_edsr = lr_tensor_cpu.unsqueeze(0).to(DEVICE) # Input for EDSR

            # 1. EDSR Prediction
            edsr_sr = model(lr_input_edsr).squeeze(0).cpu()
            edsr_sr = torch.clamp(edsr_sr, 0, 1)

            # 2. Bicubic Baseline
            # upsample_bicubic function now handles device transfer
            bicubic_sr = upsample_bicubic(lr_tensor_cpu, scale_factor)

            # Convert Tensors to NumPy for Plotting (H, W)
            def to_img_array(tensor):
                return tensor.squeeze().cpu().numpy()

            lr_img = to_img_array(lr_tensor_cpu)
            bicubic_img = to_img_array(bicubic_sr)
            edsr_img = to_img_array(edsr_sr)
            hr_img = to_img_array(hr_tensor)

            # --- Plotting ---
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            
            images = [lr_img, bicubic_img, edsr_img, hr_img]
            titles = [
                f"LR Input (x{scale_factor})", 
                f"Bicubic SR", 
                f"EDSR SR", 
                f"HR Target (Ground Truth)"
            ]
            
            for ax, img, title in zip(axes, images, titles):
                # Use 'viridis' for radar data
                ax.imshow(img, cmap='viridis', vmin=0, vmax=1, interpolation='nearest') 
                ax.set_title(title)
                ax.axis('off')

            patch_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{patch_name}_comparison.png")
            plt.suptitle(f"x{scale_factor} Super-Resolution Comparison: {patch_name}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(output_file)
            plt.close(fig)

    print(f"Visualizations saved to: {output_dir}")


def main_comparison():
    
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    
    # 1. --- Evaluate X2 ---
    model_x2 = load_model(CHECKPOINT_PATH_X2, scale_factor=2, device=DEVICE)
    
    results_x2 = calculate_metrics(model_x2, scale_factor=2)
    if results_x2:
        print("\n==============================================")
        print(f"| RESULTS SUMMARY (x{results_x2['scale']}) |")
        print("==============================================")
        print(f"| Metric  | EDSR (Ours) | Bicubic (Baseline) |")
        print("|---------|-------------|--------------------|")
        print(f"| PSNR    | {results_x2['EDSR_PSNR']:.4f} dB | {results_x2['BICUBIC_PSNR']:.4f} dB      |")
        print(f"| SSIM    | {results_x2['EDSR_SSIM']:.4f} | {results_x2['BICUBIC_SSIM']:.4f}           |")
        print("==============================================")
    
        visualize_comparison(model_x2, scale_factor=2, num_samples=NUM_VISUAL_SAMPLES)


    # 2. --- Evaluate X4 ---
    model_x4 = load_model(CHECKPOINT_PATH_X4, scale_factor=4, device=DEVICE)
    
    results_x4 = calculate_metrics(model_x4, scale_factor=4)
    if results_x4:
        print("\n==============================================")
        print(f"| RESULTS SUMMARY (x{results_x4['scale']}) |")
        print("==============================================")
        print(f"| Metric  | EDSR (Ours) | Bicubic (Baseline) |")
        print("|---------|-------------|--------------------|")
        print(f"| PSNR    | {results_x4['EDSR_PSNR']:.4f} dB | {results_x4['BICUBIC_PSNR']:.4f} dB      |")
        print(f"| SSIM    | {results_x4['EDSR_SSIM']:.4f} | {results_x4['BICUBIC_SSIM']:.4f}           |")
        print("==============================================")
        
        visualize_comparison(model_x4, scale_factor=4, num_samples=NUM_VISUAL_SAMPLES)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main_comparison()


