from torchvision import transforms
import numpy as np
from PIL import Image
import torch

def load_image_streamlit(pil_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(pil_image).unsqueeze(0)

def save_mask_array(array, path):
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(path)
