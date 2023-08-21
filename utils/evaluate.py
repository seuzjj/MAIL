from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity
import numpy as np
from skimage.metrics import mean_squared_error

def PSNR(img1, img2):
    img = img1.data.cpu().numpy().astype(np.float32)
    img_clean = img2.data.cpu().numpy().astype(np.float32)
    # img = (img-np.min(img))/(np.max(img)-np.min(img))
    # img_clean=(img_clean-np.min(img_clean))/(np.max(img_clean)-np.min(img_clean))
    data_range = (np.max(img_clean)-np.min(img_clean))
    PSNR = compare_psnr(img_clean[0, 0, :, :], img[0, 0, :, :], data_range=data_range)
    mse = mean_squared_error(img_clean[0, 0, :, :], img[0, 0, :, :])
    return PSNR


def SSIM(img1, img2):
    img = img1.data.cpu().numpy().astype(np.float32)
    img_clean = img2.data.cpu().numpy().astype(np.float32)
    # img = img/np.max(img)
    # img_clean=(img_clean-np.min(img_clean))/(np.max(img_clean)-np.min(img_clean))
    data_range = (np.max(img_clean)-np.min(img_clean))*0.6
    SSIM = structural_similarity(img_clean[0, 0, :, :], img[0, 0, :, :], data_range=data_range)
    return SSIM
