import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

from diffusers import UNet2DConditionModel
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Hint Encoder
# ------------------------------
class HintEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base.fc = nn.Linear(self.base.fc.in_features, out_dim)

    def forward(self, x):
        x = self.base(x)
        return x.unsqueeze(1)

# ------------------------------
# Self-Attention Encoder
# ------------------------------
class SelfAttentionEncoder(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        x = self.proj(x)
        out, _ = self.attn(x, x, x)
        return out

# ------------------------------
# Load Paths
# ------------------------------
mapping_path = "DIV2K_valid_HR/benchmark_ready2/image_mappings.json"
image_dir = "DIV2K_valid_HR/benchmark_ready2/"
clean_dir = "DIV2K_valid_HR/DIV2K_valid_HR/"
export_dir = "model_outputs_eval"
os.makedirs(export_dir, exist_ok=True)

# ------------------------------
# Resize transform
# ------------------------------
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()

# ------------------------------
# Load Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet2DConditionModel(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=768
).to(device)

epoch = 10

unet.load_state_dict(torch.load(f"cross_attn_denoiser_unet_epoch{epoch}.pth", map_location=device))
unet.eval()

hint_encoder = HintEncoder(out_dim=768).to(device)
hint_encoder.load_state_dict(torch.load(f"hint_encoder_epoch{epoch}.pth", map_location=device))
hint_encoder.eval()

self_attn_encoder = SelfAttentionEncoder(embed_dim=768).to(device)
self_attn_encoder.load_state_dict(torch.load(f"self_attn_encoder_epoch{epoch}.pth", map_location=device))
self_attn_encoder.eval()

# ------------------------------
# Evaluate
# ------------------------------
with open(mapping_path, 'r') as f:
    mappings = json.load(f)

results = {}

for mapping in tqdm(mappings, desc="Evaluating Trained Model"):
    noise_level = str(mapping['noise_level'])

    noisy_path = os.path.join(image_dir, mapping['noisy_image'])
    hint_path = os.path.join(image_dir, mapping['denoised_image'])
    clean_path = os.path.join(clean_dir, mapping['original_image'])

    noisy = resize(Image.open(noisy_path).convert("RGB"))
    hint = resize(Image.open(hint_path).convert("RGB"))
    clean = resize(Image.open(clean_path).convert("RGB"))

    noisy_tensor = to_tensor(noisy).unsqueeze(0).to(device)
    hint_tensor = to_tensor(hint).unsqueeze(0).to(device)
    clean_tensor = to_tensor(clean).unsqueeze(0).to(device)

    with torch.no_grad():
        hint_embed = hint_encoder(hint_tensor)
        self_embed = self_attn_encoder(noisy_tensor)
        combined_embed = torch.cat([hint_embed, self_embed], dim=1)
        timestep = torch.zeros(noisy_tensor.shape[0], dtype=torch.long).to(device)
        output = unet(noisy_tensor, timestep=timestep, encoder_hidden_states=combined_embed).sample

    output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    clean_np = clean_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    psnr_val = psnr_metric(clean_np, output_np, data_range=1.0)
    ssim_val = ssim_metric(clean_np, output_np, channel_axis=-1, data_range=1.0)
    recon_error = np.mean((output_np - clean_np) ** 2)

    if noise_level not in results:
        results[noise_level] = {'psnr': [], 'ssim': [], 'reconstruction_error': []}

    results[noise_level]['psnr'].append(psnr_val)
    results[noise_level]['ssim'].append(ssim_val)
    results[noise_level]['reconstruction_error'].append(recon_error)

    # # ------------------------------
    # # Save Result Image
    # # ------------------------------
    # # Create subdir per noise level
    # save_subdir = os.path.join(export_dir, f"noise_{noise_level}")
    # os.makedirs(save_subdir, exist_ok=True)

    # # Save output image with descriptive name
    # filename_base = os.path.splitext(os.path.basename(mapping['original_image']))[0]
    # save_path = os.path.join(save_subdir, f"{filename_base}_denoised.png")
    # result_image = torch.cat([noisy_tensor, hint_tensor, output, clean_tensor], dim=0)
    # save_image(result_image, save_path, nrow=1)

# ------------------------------
# Print Average Results
# ------------------------------
print("\nEvaluation Results for Trained Model:")
for noise_level in sorted(results.keys(), key=lambda x: float(x)):
    avg_psnr = np.mean(results[noise_level]['psnr'])
    avg_ssim = np.mean(results[noise_level]['ssim'])
    avg_recon_error = np.mean(results[noise_level]['reconstruction_error'])
    print(f"Noise Level: {noise_level} -> "
          f"Average PSNR: {avg_psnr:.2f} dB, "
          f"Average SSIM: {avg_ssim:.4f}, "
          f"Reconstruction Error (MSE): {avg_recon_error:.6f}")
