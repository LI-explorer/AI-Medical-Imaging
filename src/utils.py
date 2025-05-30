import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def add_noise(img, noise_type="gaussian", std=0.05):
    if noise_type == "gaussian":
        noisy = img + std * np.random.randn(*img.shape)
        return np.clip(noisy, 0, 1)
    return img

def evaluate(img_true, img_pred):
    psnr = peak_signal_noise_ratio(img_true, img_pred, data_range=img_true.max() - img_true.min())
    ssim = structural_similarity(img_true, img_pred, data_range=img_true.max() - img_true.min())
    return psnr, ssim
