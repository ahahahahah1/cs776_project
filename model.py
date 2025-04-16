import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from diffusers import UNet2DConditionModel
from torchvision.utils import save_image
from PIL import Image
import os
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

# For SSIM loss using pytorch_msssim
import pytorch_msssim

# ------------------------------
# Dataset Definition
# ------------------------------
class NoisyDenoiseDataset(Dataset):
    def __init__(self, mapping_json_path, image_dir, clean_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.clean_dir = clean_dir
        self.transform = transform

        with open(mapping_json_path, 'r') as f:
            self.mappings = json.load(f)

    def __len__(self):
        return len(self.mappings)

    def __getitem__(self, idx):
        mapping = self.mappings[idx]
        noisy_path = os.path.join(self.image_dir, mapping['noisy_image'])
        tv_hint_path = os.path.join(self.image_dir, mapping['denoised_image'])
        clean_path = os.path.join(self.clean_dir, mapping['original_image'])

        noisy = Image.open(noisy_path).convert("RGB")
        tv_hint = Image.open(tv_hint_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        if self.transform:
            noisy = self.transform(noisy)
            tv_hint = self.transform(tv_hint)
            clean = self.transform(clean)

        return noisy, tv_hint, clean, mapping['original_image']

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
        return x.unsqueeze(1)  # Add a singleton dimension for later concatenation

# ------------------------------
# Self-Attention Encoder (Patch-based)
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
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        
        # Extract non-overlapping patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # Rearrange patches for attention: [B, num_patches, patch_vector]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        x = self.proj(x)
        out, _ = self.attn(x, x, x)
        return out

# ------------------------------
# Combined Loss Function: MSE + Perceptual + SSIM
# ------------------------------
# Setup the perceptual loss network using VGG16 (features up to relu3_3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(weights='IMAGENET1K_V1').features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

def ssim_loss(pred, target):
    # 1 - SSIM gives us a loss value
    return 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_net, perceptual_weight=0.1, ssim_weight=0.5):
        super().__init__()
        self.perceptual_net = perceptual_net
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target, hint=None):
        # Pixel-wise MSE loss
        loss_mse = self.mse(pred, target)
        
        # Perceptual loss on VGG extracted features
        feat_pred = self.perceptual_net(pred)
        feat_target = self.perceptual_net(target)
        loss_percep = self.mse(feat_pred, feat_target)
        
        # Structural loss (SSIM)
        loss_ssim = ssim_loss(pred, target)
        
        total_loss = loss_mse + self.perceptual_weight * loss_percep + self.ssim_weight * loss_ssim

        if hint is not None:
            # Optionally add an L1 loss with the hint image
            loss_hint = F.l1_loss(pred, hint)
            total_loss += 0.05 * loss_hint

        return total_loss

# ------------------------------
# Configurations and Transforms (256x256 Images)
# ------------------------------
# Paths for validation set
valid_mapping_json_path = "DIV2K_valid_HR/benchmark_ready2/image_mappings.json"
valid_image_dir = "DIV2K_valid_HR/benchmark_ready2/"
valid_clean_dir = "DIV2K_valid_HR/DIV2K_valid_HR/"

# Paths for training set
train_mapping_json_path = "./DIV2K_train_HR/benchmark_ready/image_mappings.json" 
train_image_dir = "./DIV2K_train_HR/benchmark_ready/"
train_clean_dir = "./DIV2K_train_HR/DIV2K_train_HR/"

export_dir = "validation_outputs"
os.makedirs(export_dir, exist_ok=True)

batch_size = 4
epochs = 10
learning_rate = 1e-5

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ------------------------------
# Datasets and Dataloaders
# ------------------------------
# Training dataset using training paths
train_dataset = NoisyDenoiseDataset(train_mapping_json_path, train_image_dir, train_clean_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset using validation paths
val_dataset = NoisyDenoiseDataset(valid_mapping_json_path, valid_image_dir, valid_clean_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1)

# ------------------------------
# Model Setup
# ------------------------------
# Adjust UNet sample_size to 256 to match input dimensions
small_unet = UNet2DConditionModel(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=768
).to(device)
unet = small_unet

hint_encoder = HintEncoder(out_dim=768).to(device)
self_attn_encoder = SelfAttentionEncoder(embed_dim=768).to(device)

optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(hint_encoder.parameters()) + list(self_attn_encoder.parameters()),
    lr=learning_rate
)

combined_loss_fn = CombinedLoss(perceptual_net=vgg, perceptual_weight=0.1, ssim_weight=0.5).to(device)

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(epochs):
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
    unet.train()
    hint_encoder.train()
    self_attn_encoder.train()

    for noisy, tv_hint, clean, _ in train_bar:
        noisy, tv_hint, clean = noisy.to(device), tv_hint.to(device), clean.to(device)

        with torch.no_grad():
            # Generate embeddings from hint and self-attention encoders
            hint_embed = hint_encoder(tv_hint)
            self_embed = self_attn_encoder(noisy)
            combined_embed = torch.cat([hint_embed, self_embed], dim=1)

        # UNet requires a timestep; here we use a dummy timestep (zero)
        timestep = torch.zeros(noisy.shape[0], dtype=torch.long).to(device)
        output = unet(noisy, timestep=timestep, encoder_hidden_states=combined_embed).sample

        loss = combined_loss_fn(output, clean, hint=tv_hint)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f}")

    # ------------------------------
    # Validation and Visualization
    # ------------------------------
    unet.eval()
    hint_encoder.eval()
    self_attn_encoder.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
        for i, (noisy, tv_hint, clean, filename) in enumerate(val_bar):
            noisy, tv_hint, clean = noisy.to(device), tv_hint.to(device), clean.to(device)

            hint_embed = hint_encoder(tv_hint)
            self_embed = self_attn_encoder(noisy)
            combined_embed = torch.cat([hint_embed, self_embed], dim=1)

            timestep = torch.zeros(noisy.shape[0], dtype=torch.long).to(device)
            output = unet(noisy, timestep=timestep, encoder_hidden_states=combined_embed).sample

            # Save concatenated images: noisy, hint, output, and clean
            save_path = os.path.join(export_dir, f"epoch_{epoch+1}_{filename[0]}")
            save_image(torch.cat([noisy, tv_hint, output, clean], dim=0), save_path, nrow=1)

            # Calculate PSNR and SSIM
            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            clean_np = clean.squeeze().cpu().numpy().transpose(1, 2, 0)
            psnr_val = psnr_metric(clean_np, output_np, data_range=1.0)
            ssim_val = ssim_metric(clean_np, output_np, channel_axis=-1, data_range=1.0)
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"Epoch [{epoch+1}] Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

# ------------------------------
# Save Models
# ------------------------------
torch.save(unet.state_dict(), "cross_attn_denoiser_unet.pth")
torch.save(hint_encoder.state_dict(), "hint_encoder.pth")
torch.save(self_attn_encoder.state_dict(), "self_attn_encoder.pth")
