import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_efficientnetb0_12bands.pth"

# ------------------------------
# Load model
# ------------------------------
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=12,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------------------
# Preprocessing
# ------------------------------
global_min = np.array([-1393., -1169., -722., -684., -412., -335., -251., 64., -9999., 8., 10., 0.], dtype=np.float32)
global_max = np.array([6568., 9659., 11368., 12041., 15841., 15252., 14647., 255., 4245., 4287., 100., 111.], dtype=np.float32)

def preprocess_image(file_path):
    with rasterio.open(file_path) as src:
        img = src.read().astype(np.float32)  # (12,H,W)
    # normalize each band
    for c in range(img.shape[0]):
        img[c] = np.clip(img[c], global_min[c], global_max[c])
        img[c] = (img[c] - global_min[c]) / (global_max[c] - global_min[c] + 1e-8)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)  # (1,12,H,W)
    return img_tensor, img

def load_mask(mask_path):
    mask = plt.imread(mask_path)
    mask = (mask > 0).astype(np.float32)
    return mask

def predict_mask(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        output = (output > threshold).float()
    return output[0,0].cpu().numpy()  # (H,W)

# ------------------------------
# Example usage
# ------------------------------
image_path = "26.tif"    # غيّره للصورة اللي بدك تجربها
mask_path  = "26.png"    # ماسكها الأصلي

image_tensor, img_np = preprocess_image(image_path)
pred_mask = predict_mask(model, image_tensor)
orig_mask = load_mask(mask_path)

# RGB view
rgb = img_np[:3].transpose(1,2,0)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(orig_mask, cmap="gray")
plt.title("Original Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()