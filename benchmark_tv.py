import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
from torchvision import transforms

# Define the paths to the mapping file and directories:
valid_mapping_json_path = "DIV2K_valid_HR/benchmark_ready2/image_mappings.json"
denoised_dir = "DIV2K_valid_HR/benchmark_ready2/"     # TV denoised outputs are stored here
clean_dir = "DIV2K_valid_HR/DIV2K_valid_HR/"            # Original clean images

# Load the JSON mapping file
with open(valid_mapping_json_path, 'r') as f:
    mappings = json.load(f)

# Define a transform to resize the images to 256x256
resize_transform = transforms.Resize((256, 256))

# Initialize a results dictionary to store scores per noise level.
results = {}

# Iterate over all entries in the mapping file
for mapping in tqdm(mappings, desc="Evaluating TV Denoising"):
    noise_level = mapping['noise_level']  # This value is expected to be a float (or string)

    # Construct full paths to the images
    denoised_path = os.path.join(denoised_dir, mapping['denoised_image'])
    original_path = os.path.join(clean_dir, mapping['original_image'])
    
    # Load images using PIL and convert to RGB
    denoised_img = Image.open(denoised_path).convert("RGB")
    original_img = Image.open(original_path).convert("RGB")
    
    # Resize images to 256x256
    denoised_img = resize_transform(denoised_img)
    original_img = resize_transform(original_img)
    
    # Convert images to numpy arrays and normalize to [0,1]
    denoised_np = np.array(denoised_img).astype(np.float32) / 255.0
    original_np = np.array(original_img).astype(np.float32) / 255.0
    
    # Compute PSNR and SSIM metrics
    psnr_val = psnr_metric(original_np, denoised_np, data_range=1.0)
    ssim_val = ssim_metric(original_np, denoised_np, channel_axis=-1, data_range=1.0)
    
    # Compute reconstruction error (MSE) on a per-pixel basis
    reconstruction_error = np.mean((original_np - denoised_np) ** 2)
    
    # Group the results by noise level (converted to string for consistency)
    key = str(noise_level)
    if key not in results:
        results[key] = {'psnr': [], 'ssim': [], 'reconstruction_error': []}
    results[key]['psnr'].append(psnr_val)
    results[key]['ssim'].append(ssim_val)
    results[key]['reconstruction_error'].append(reconstruction_error)

# Now compute and print average PSNR, SSIM, and Reconstruction Error for each noise level
print("\nBenchmarking TV Denoising results (images resized to 256x256):")
for noise_level in sorted(results.keys(), key=lambda x: float(x)):
    avg_psnr = np.mean(results[noise_level]['psnr'])
    avg_ssim = np.mean(results[noise_level]['ssim'])
    avg_recon_error = np.mean(results[noise_level]['reconstruction_error'])
    print(f"Noise Level: {noise_level} -> Average PSNR: {avg_psnr:.2f} dB, "
          f"Average SSIM: {avg_ssim:.4f}, Reconstruction Error (MSE): {avg_recon_error:.4f}")
