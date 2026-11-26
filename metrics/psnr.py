import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, \
        (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calc_psnr_and_ssim(img1, img2):
    """Calculate PSNR and SSIM for images.
        img1: ndarray, range [0, 255]
        img2: ndarray, range [0, 255]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    psnr = calculate_psnr(img1, img2)
    ssim = compare_ssim(img1, img2,
                        data_range=255,
                        multichannel=True,
                        win_size=65,
                        channel_axis=2)

    return psnr, ssim





