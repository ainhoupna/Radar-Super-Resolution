import os
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
from PIL import Image

# ================= CONFIGURATION =================
INPUT_DIR = "georef_dana_results_x4"
INPUT_DIR = "georef_dana_results_x2"
OUTPUT_GIF = "dana_zaragoza_comparison_x2.gif"
FPS = 0.2 
# =================================================

def main():
    # 1. Get Image List
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')]
    files.sort()
    
    if not files:
        print("No images found in the input directory.")
        return

    print(f"Creating animation from {len(files)} images...")
    
    images = []
    target_shape = None
    
    for filename in tqdm(files, desc="Reading and resizing images"):
        img_path = os.path.join(INPUT_DIR, filename)
        img = imageio.imread(img_path)
        
        if target_shape is None:
            target_shape = img.shape[:2] # (H, W)
        
        if img.shape[:2] != target_shape:
            # Resize using PIL to match the first frame's dimensions
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            img = np.array(img_pil)
            
        images.append(img)

    # 2. Save as GIF (looping)
    print(f"Saving GIF to {OUTPUT_GIF}...")
    imageio.mimsave(OUTPUT_GIF, images, duration=1/FPS, loop=0)

    # 3. Save as MP4
    print(f"Saving MP4 to {OUTPUT_MP4}...")
    try:
        # Using FFMPEG for MP4
        imageio.mimsave(OUTPUT_MP4, images, fps=FPS, format='FFMPEG', quality=8)
    except Exception as e:
        print(f"Error saving MP4: {e}")

    print("Animation creation complete.")

if __name__ == "__main__":
    main()
