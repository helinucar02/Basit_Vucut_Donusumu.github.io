import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def tps_deform(image: Image.Image, mask: Image.Image, mode="slim"):
    img = np.array(image)
    h, w = img.shape[:2]

    mask_np = np.array(mask.convert("L"))
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    key_regions = {
        "bel": y + int(bh * 0.60),
        "kalça": y + int(bh * 0.75),
        "göğüs": y + int(bh * 0.40),
        "omuz": y + int(bh * 0.25)
    }

    src_pts = []
    dst_pts = []

    factor = -0.15 if mode == "slim" else 0.15

    for name, cy in key_regions.items():
        for dx in [-40, 0, 40]:
            src_pts.append([x + bw//2 + dx, cy])
            dst_pts.append([x + bw//2 + int(dx * (1 + factor)), cy])

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(dst_pts[None, :, :], src_pts[None, :, :], matches)

    warped = tps.warpImage(img)
    blended = blend_edges(warped, mask)
    return blended
def blend_edges(image: np.ndarray, mask: Image.Image):
    image_np = np.array(image)


    alpha = np.array(mask.convert("L")) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)


    alpha_3c = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)


    blurred_body = gaussian_filter(image_np * alpha_3c, sigma=(3, 3, 0))

    blended = (blurred_body + image_np * (1 - alpha_3c)).astype(np.uint8)
    return blended