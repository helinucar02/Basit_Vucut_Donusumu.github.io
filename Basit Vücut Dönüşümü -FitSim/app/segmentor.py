import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from app.u2net_model import U2NET

def load_model():
    model = U2NET(3, 1)
    model.load_state_dict(torch.load("u2net/u2net.pth", map_location="cpu"))
    model.eval()
    return model

def segment(image: Image.Image, model) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = d1[0][0].cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = (pred * 255).astype(np.uint8)
        mask = cv2.resize(mask, image.size)
    return mask