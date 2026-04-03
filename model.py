import torch
import segmentation_models_pytorch as smp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "unet_efficientnetb0_12bands.pth"

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