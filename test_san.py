import torch
from models.san.san import SAN

class Args:
    n_resgroups = 10
    n_resblocks = 10
    n_feats = 64
    reduction = 16
    scale = [2]
    res_scale = 1.0
    rgb_range = 1.0
    n_colors = 1

def test():
    try:
        model = SAN(Args())
        x = torch.randn(1, 1, 48, 48)
        y = model(x)
        print(f"Success! Input: {x.shape}, Output: {y.shape}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
