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
from piq import ssim as ssim_loss  # Differentiable SSIM para entrenamiento

# ================= CONFIGURACIÓN =================
TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

# EXPERIMENT 5: Deeper Network (Sin AMP)
SCALE_FACTOR = 2       
N_RESBLOCKS = 32       
N_FEATS = 64           
RES_SCALE = 0.1        

BATCH_SIZE = 64        
LEARNING_RATE = 5e-5   
EPOCHS = 500           
PATIENCE = 35           
ALPHA_SSIM = 0.2      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_ROOT_DIR = "checkpoints_edsr"
MLFLOW_EXPERIMENT_NAME = "AEMET_Radar_EDSR_Optimization"
# =================================================

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

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
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

        self.head = nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2)
        
        m_body = [ResBlock(n_feats, kernel_size, act=act, res_scale=RES_SCALE) for _ in range(n_resblocks)]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        self.body = nn.Sequential(*m_body)
        
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=kernel_size//2),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

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

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"edsr_x{SCALE_FACTOR}_depth32_noAMP_{timestamp}"
    CURRENT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT_DIR, RUN_NAME)
    os.makedirs(CURRENT_CHECKPOINT_DIR, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "scale": SCALE_FACTOR,
            "res_blocks": N_RESBLOCKS,
            "features": N_FEATS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "alpha_ssim": ALPHA_SSIM,
            "mixed_precision": False, # Ahora sí es real
            "augmentation": "dihedral"
        })

        train_loader = DataLoader(RadarDataset(TRAIN_DIR, SCALE_FACTOR, augment=True), 
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(RadarDataset(VAL_DIR, SCALE_FACTOR, augment=False), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = EDSR({'n_resblocks': N_RESBLOCKS, 'n_feats': N_FEATS, 'scale': SCALE_FACTOR}).to(DEVICE)
        criterion_l1 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=PATIENCE, delta=0.01)

        best_psnr = 0.0

        for epoch in range(EPOCHS):
            model.train()
            train_loss_accum = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for lr, hr in loop:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                
                # Paso de entrenamiento estándar (FP32 completo)
                optimizer.zero_grad()
                sr = model(lr)
                l1 = criterion_l1(sr, hr)
                ssim_val = ssim_loss(torch.clamp(sr, 0, 1), hr, data_range=1.0)
                loss = l1 + ALPHA_SSIM * (1.0 - ssim_val)
                
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                loop.set_postfix(loss=loss.item(), ssim=ssim_val.item())

            model.eval()
            val_psnr, val_ssim = 0, 0
            with torch.no_grad():
                for lr, hr in val_loader:
                    lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                    sr = torch.clamp(model(lr), 0, 1)
                    
                    val_psnr += calculate_psnr(sr, hr).item()
                    val_ssim += ssim_loss(sr, hr, data_range=1.0).item()
            
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
                print(f"--> ¡Nuevo mejor modelo guardado! ({best_psnr:.2f} dB)")

            scheduler.step()
            early_stopping(avg_psnr)
            if early_stopping.early_stop:
                print(f"[INFO] Early stopping activado en la época {epoch+1}.")
                break

    print("Entrenamiento Experimento 5 (Sin AMP) completado.")

if __name__ == "__main__":
    main()