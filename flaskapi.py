from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import segmentation_models_pytorch as smp
import rasterio
from PIL import Image
import io
import numpy as np
from skimage.morphology import opening, closing, disk

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_efficientnetb0_12bands.pth"

# ------------------------------
# Model
# ------------------------------
def get_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=12,
        classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = get_model()

# ------------------------------
# Preprocessing
# ------------------------------
# Use same global min/max as training
global_min = np.array([-1393., -1169., -722., -684., -412., -335., -251., 64., -9999., 8., 10., 0.], dtype=np.float32)
global_max = np.array([6568., 9659., 11368., 12041., 15841., 15252., 14647., 255., 4245., 4287., 100., 111.], dtype=np.float32)

def preprocess_image(file):

    with rasterio.open(file) as src:
        img = src.read().astype(np.float32)  # (12,H,W)

    for c in range(img.shape[0]):
        img[c] = np.clip(img[c], global_min[c], global_max[c])
        img[c] = (img[c] - global_min[c]) / (global_max[c] - global_min[c] + 1e-8)
        img[c] = np.clip(img[c], 0, 1)
        img = np.nan_to_num(img)

    img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    return img_tensor

def predict_mask(model, image_tensor, threshold=0.4):
    """Predict binary mask with optional post-processing."""
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        output = (output > threshold).float().cpu().numpy()[0,0]  # (H,W)


    # Post-processing: remove small noise
    output = closing(opening(output, disk(1)), disk(1))

    # Debug print
    print("Pred mask min/max:", output.min(), output.max(), "Unique values:", np.unique(output))

    return output

def mask_to_image(mask_array):
    """Convert mask numpy array (0/1) to PIL Image."""
    mask_img = (mask_array * 255).astype(np.uint8)
    return Image.fromarray(mask_img)

# ------------------------------
# Flask API
# ------------------------------

@app.route("/")
def home():
    return render_template('index.html', index=5)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_tensor = preprocess_image(file)
    pred_mask = predict_mask(model, image_tensor, threshold=0.4)
    mask_img = mask_to_image(pred_mask)

    img_io = io.BytesIO()
    mask_img.save(img_io, 'PNG')
    img_io.seek(0)
    return app.response_class(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)