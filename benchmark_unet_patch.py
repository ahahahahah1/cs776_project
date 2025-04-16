import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

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
        # Unfold into non-overlapping patches (each patch is patch_size x patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        x = self.proj(x)
        out, _ = self.attn(x, x, x)
        return out

# ------------------------------
# Set Paths and Directories
# ------------------------------
mapping_path = "DIV2K_valid_HR/benchmark_ready2/image_mappings.json"
noisy_dir = "DIV2K_valid_HR/benchmark_ready2/"       # Contains both TV-model hint and noisy images.
clean_dir = "DIV2K_valid_HR/DIV2K_valid_HR/"          # Clean (ground-truth) images.
export_dir = "model_outputs_eval"
os.makedirs(export_dir, exist_ok=True)

# ------------------------------
# Transforms
# ------------------------------
# For patch-based processing we work with 512x512 images.
resize512 = transforms.Resize((512, 512))
# For final evaluation we resize directly to 256x256.
resize256 = transforms.Resize((256, 256))
# Conversion from PIL to tensor.
to_tensor = transforms.ToTensor()
# Conversion from tensor back to PIL image.
to_pil = transforms.ToPILImage()

# ------------------------------
# Load Model and Encoders
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 10  # Set to the epoch you want to evaluate.

unet = UNet2DConditionModel(
    sample_size=256,  # Final pass works on 256x256 images.
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=768
).to(device)
unet.load_state_dict(torch.load(f"cross_attn_denoiser_unet_epoch{epoch}.pth", map_location=device))
unet.eval()

hint_encoder = HintEncoder(out_dim=768).to(device)
hint_encoder.load_state_dict(torch.load(f"hint_encoder_epoch{epoch}.pth", map_location=device))
hint_encoder.eval()

self_attn_encoder = SelfAttentionEncoder(embed_dim=768).to(device)
self_attn_encoder.load_state_dict(torch.load(f"self_attn_encoder_epoch{epoch}.pth", map_location=device))
self_attn_encoder.eval()

# ------------------------------
# Helper: Split 512x512 image into four 256x256 patches.
# ------------------------------
def split_into_patches(img):
    """
    Splits a 512x512 PIL image into four non-overlapping 256x256 patches.
    Order: top-left, top-right, bottom-left, bottom-right.
    """
    patches = []
    patches.append(img.crop((0, 0, 256, 256)))      # Top-left
    patches.append(img.crop((256, 0, 512, 256)))     # Top-right
    patches.append(img.crop((0, 256, 256, 512)))      # Bottom-left
    patches.append(img.crop((256, 256, 512, 512)))    # Bottom-right
    return patches

# ------------------------------
# Set Number of Patch-Based Passes
# ------------------------------
num_passes = 4  # Change this value to the desired number of iterative patch-based passes.

# ------------------------------
# Evaluation Loop (Iterative Passes + Final Pass)
# ------------------------------
with open(mapping_path, 'r') as f:
    mappings = json.load(f)

# Dictionary to group final metrics by noise level.
results = {}

for mapping in tqdm(mappings, desc="Evaluating Model with Iterative Passes"):
    noise_level = str(mapping['noise_level'])
    
    # Load the full images and resize them to 512x512 (for patch-based processing).
    noisy_full = resize512(Image.open(os.path.join(noisy_dir, mapping['noisy_image'])).convert("RGB"))
    tv_hint_full = resize512(Image.open(os.path.join(noisy_dir, mapping['denoised_image'])).convert("RGB"))
    clean_full = resize512(Image.open(os.path.join(clean_dir, mapping['original_image'])).convert("RGB"))
    
    # Initialize the hint with the TV-model output.
    current_hint = tv_hint_full  # PIL image (512x512)
    
    # Iterative patch-based passes.
    for pass_idx in range(num_passes):
        # Split both the noisy image and the current hint into 256x256 patches.
        noisy_patches = split_into_patches(noisy_full)
        hint_patches = split_into_patches(current_hint)
        patch_outputs = []
        
        # Process each patch.
        for n_patch, h_patch in zip(noisy_patches, hint_patches):
            # Convert patches to tensors and add batch dimension.
            n_tensor = to_tensor(n_patch).unsqueeze(0).to(device)
            h_tensor = to_tensor(h_patch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                h_embed = hint_encoder(h_tensor)
                s_embed = self_attn_encoder(n_tensor)
                combined_embed = torch.cat([h_embed, s_embed], dim=1)
                timestep = torch.zeros(n_tensor.shape[0], dtype=torch.long).to(device)
                out_patch = unet(n_tensor, timestep=timestep, encoder_hidden_states=combined_embed).sample
            patch_outputs.append(out_patch.squeeze(0))  # shape [3,256,256]
        
        # Stitch the four patch outputs into a full 512x512 output.
        top_row = torch.cat([patch_outputs[0], patch_outputs[1]], dim=2)
        bottom_row = torch.cat([patch_outputs[2], patch_outputs[3]], dim=2)
        full_out_tensor = torch.cat([top_row, bottom_row], dim=1)  # shape [3,512,512]
        
        # Update the hint for the next pass: convert the tensor output to PIL.
        current_hint = to_pil(full_out_tensor.cpu())
        
        # Optionally, you could save intermediate pass outputs.
        save_path = os.path.join(export_dir, f"noise_{noise_level}", f"{os.path.splitext(os.path.basename(mapping['original_image']))[0]}_pass{pass_idx+1}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(full_out_tensor.unsqueeze(0), save_path)
    
    # ------------------------------
    # Final Pass: Full-Image Processing Without Patching.
    # ------------------------------
    # Resize the noisy full image and the final iterative hint (current_hint) directly to 256x256.
    noisy_256 = to_tensor(resize256(noisy_full)).unsqueeze(0).to(device)
    # Convert current_hint (PIL at 512x512) to tensor then resize to 256x256.
    current_hint_tensor = to_tensor(current_hint).unsqueeze(0).to(device)
    hint_256 = F.interpolate(current_hint_tensor, size=(256,256), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        h_embed_final = hint_encoder(hint_256)
        s_embed_final = self_attn_encoder(noisy_256)
        combined_embed_final = torch.cat([h_embed_final, s_embed_final], dim=1)
        timestep_final = torch.zeros(noisy_256.shape[0], dtype=torch.long).to(device)
        final_output = unet(noisy_256, timestep=timestep_final, encoder_hidden_states=combined_embed_final).sample
        # final_output is 256x256.
        
    # Convert clean image to 256x256 for metric evaluation.
    clean_256 = to_tensor(resize256(clean_full)).unsqueeze(0).to(device)
    
    # Compute metrics on final_output vs. clean_256.
    output_np = final_output.squeeze().cpu().numpy().transpose(1,2,0)
    clean_np = clean_256.squeeze().cpu().numpy().transpose(1,2,0)
    psnr_val = psnr_metric(clean_np, output_np, data_range=1.0)
    ssim_val = ssim_metric(clean_np, output_np, channel_axis=-1, data_range=1.0)
    recon_error = np.mean((output_np - clean_np)**2)
    
    # Store metrics per noise level.
    if noise_level not in results:
        results[noise_level] = {'psnr': [], 'ssim': [], 'reconstruction_error': []}
    results[noise_level]['psnr'].append(psnr_val)
    results[noise_level]['ssim'].append(ssim_val)
    results[noise_level]['reconstruction_error'].append(recon_error)
    
    # Save the final 256x256 output.
    final_save_subdir = os.path.join(export_dir, f"noise_{noise_level}")
    os.makedirs(final_save_subdir, exist_ok=True)
    final_save_path = os.path.join(final_save_subdir, f"{os.path.splitext(os.path.basename(mapping['original_image']))[0]}_final.png")
    save_image(final_output, final_save_path)

# ------------------------------
# Print Final Average Results (Metrics Computed on 256x256 Final Outputs)
# ------------------------------
print("\nFinal Evaluation Results (Final pass on 256x256 images):")
for noise_level in sorted(results.keys(), key=lambda x: float(x)):
    avg_psnr = np.mean(results[noise_level]['psnr'])
    avg_ssim = np.mean(results[noise_level]['ssim'])
    avg_recon_error = np.mean(results[noise_level]['reconstruction_error'])
    print(f"Noise Level: {noise_level} -> "
          f"Average PSNR: {avg_psnr:.2f} dB, "
          f"Average SSIM: {avg_ssim:.4f}, "
          f"Reconstruction Error (MSE): {avg_recon_error:.6f}")
