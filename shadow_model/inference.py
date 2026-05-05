import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from .architecture import FullModel   # imports from architecture.py
import gdown  # pip install gdown
import os

# ── CFG must match what you used during training ──────────────────
_CFG = {
    'img_h': 256, 'img_w': 256,
    'embed_dim': 256, 'num_heads': 8,
    'K_superpixels': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

_model = None   # lazy-loaded singleton

# def load_model(checkpoint_path):
#     """Load the trained CTU-Net+ model from a .pth checkpoint."""
#     global _model
#     _model = FullModel(_CFG).to(_CFG['device'])
#     ckpt = torch.load(checkpoint_path,
#                       map_location=_CFG['device'],
#                       weights_only=False)
#     _model.load_state_dict(ckpt['model'])
#     _model.eval()
#     print(f"Model loaded. (Best BER={ckpt['ber']:.3f}  RMSE={ckpt['rmse']:.4f})")
#     return _model

import gdown  # pip install gdown

# Your Google Drive file ID from Step 3
GDRIVE_FILE_ID = '1DuFEoCc6J-6uEvxCEp0IIBnBjEbqlBf5'

def load_model(checkpoint_path=None):
    global _model

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pth'
        )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Auto-download if not present
    if not os.path.exists(checkpoint_path):
        print("Model not found locally. Downloading from Google Drive...")
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        gdown.download(url, checkpoint_path, quiet=False)
        print("Download complete.")

    _model = FullModel(_CFG).to(_CFG['device'])
    ckpt = torch.load(checkpoint_path,
                      map_location=_CFG['device'],
                      weights_only=False)
    _model.load_state_dict(ckpt['model'])
    _model.eval()
    print(f"Model loaded. (Best BER={ckpt['ber']:.3f}  RMSE={ckpt['rmse']:.4f})")
    return _model

def remove_shadow(image_input, checkpoint_path=None):
    """
    Remove shadow from an image.

    Args:
        image_input : str (file path) or PIL.Image or numpy array (H,W,3) uint8
        checkpoint_path : str, only needed on first call

    Returns:
        dict with keys:
            'shadow_free' : PIL.Image  (shadow removed)
            'mask'        : PIL.Image  (detected shadow mask, grayscale)
            'original'    : PIL.Image  (input resized to 256x256)
    """
    global _model
    if _model is None:
        if checkpoint_path is None:
            raise ValueError("Provide checkpoint_path on the first call.")
        load_model(checkpoint_path)

    # ── Accept path, PIL, or numpy ────────────────────────────────
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
    else:
        img = image_input.convert('RGB')

    # ── Preprocess ────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((_CFG['img_h'], _CFG['img_w'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    inp = tf(img).unsqueeze(0).to(_CFG['device'])

    # ── Inference ─────────────────────────────────────────────────
    with torch.no_grad():
        out = _model(inp)

    # ── Post-process → PIL images ─────────────────────────────────
    def to_pil(tensor):
        arr = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    shadow_free = to_pil(out['refined'])
    mask_arr    = (out['mask_prob'].squeeze().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    mask_pil    = Image.fromarray(mask_arr, mode='L')
    original    = img.resize((_CFG['img_w'], _CFG['img_h']))

    return {
        'shadow_free': shadow_free,
        'mask':        mask_pil,
        'original':    original,
    }