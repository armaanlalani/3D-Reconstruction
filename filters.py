import numpy as np

from config import SIGMA, \
                   FILTER_ITERATIONS, \
                   POST_WINDOW_SIZE, \
                   SMOOTHING_THRESHOLD


def compute_kernel(sigma=SIGMA):
    # Kernel rule of thumb:  2 times stddev on each side, rounded to odd integer
    ## TODO: Potentially mult sigma by 3, then compute kernel, then cut outside off
    ## https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    size = 2 * int(2 * sigma + 0.5) + 1
    kernel = np.linspace(-(size//2), size//2, size)
    kernel = np.array([gaussian_1D(i, sigma) for i in kernel])
    kernel = np.outer(kernel.T, kernel.T)
    # Normalize for value of 1 at the center
    kernel = kernel / kernel.max()
    return kernel


#
# Definition of 1D Gaussian
#
def gaussian_1D(x, stddev):
    denom = 1 / np.sqrt(2 * np.pi) * stddev
    return denom * np.exp(-(x ** 2) / (2 * stddev ** 2))


#
# Convolve a generic kernel with an image
#
def convolve(im, kern):
    im_h, im_w = im.shape[0], im.shape[1]
    kern_h, kern_w = kern.shape[0], kern.shape[1]
    filtered = np.zeros(im.shape)

    # Define outer padding
    pad_h, pad_w = int((kern_h - 1) / 2), int((kern_w - 1) / 2)
    pad_im = np.zeros((im_h + 2 * pad_h, im_w + 2 * pad_w))
    pad_im[pad_h:pad_im.shape[0] - pad_h, pad_w:pad_im.shape[1] - pad_w] = im

    for i in range(im_h):
        for j in range(im_w):
            filtered[i, j] = np.sum(kern * pad_im[i:i + kern_h, j:j + kern_w])
    filtered /= kern_h * kern_w

    return filtered


def gaussian_filter(image):
    print("Filtering image...")
    kernel = compute_kernel(SIGMA)
    for i in range(FILTER_ITERATIONS):
        image = convolve(image, kernel)
    return image


def post_smoothing(disp_map, window_size=POST_WINDOW_SIZE):
    h, w = disp_map.shape

    i, j = 0, 0
    while i < h:
        while j < w:
            xl, xr = max(0, i - window_size // 2), min(h - 1, i + window_size // 2)
            yt, yb = max(0, j - window_size // 2), min(w - 1, j + window_size // 2)
            window = disp_map[xl:xr, yt:yb]

            avg = np.ma.average(window, weights=window.astype(bool))
            if not np.isnan(avg):
                window[:, :] = avg

            if disp_map[i, j] > SMOOTHING_THRESHOLD:
                disp_map[i, j] = SMOOTHING_THRESHOLD

            j += window_size

        j = 0    
        i += window_size
    return disp_map
