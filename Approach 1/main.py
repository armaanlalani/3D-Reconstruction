import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import IM1, IM2
from filters import gaussian_filter, post_smoothing
from utils import load_image, reshape_pts, plot_point_cloud
from correlation import get_src_and_dest, get_disp_map, create_point_cloud, post_process_disp_map, get_window_matching_disp_map


def process_image(im):
    im = load_image(im)
    smooth_im = gaussian_filter(im)
    return smooth_im


def main():
    # Load image and apply gaussian filter
    im1_rgb = load_image(IM1, color=True)
    im1, im2 = process_image(IM1), process_image(IM2)

    # METHOD 1: Find src and dest points using SIFT, then compute disparity x_l - x_r for each match
    # Uncomment below
    # src, dest, _, _, _ = get_src_and_dest(im1, im2)
    # src, dest = reshape_pts(src), reshape_pts(dest)
    # disp_map = get_disp_map(im1_rgb, src, dest)

    # METHOD 2: Sliding window approach using the fact that every match must be on same line
    # Minimum sum of absolute differences is the disparity for each point
    # Uncomment below
    disp_map = get_window_matching_disp_map(im1, im2)

    # Common to both methods
    disp_map = post_process_disp_map(disp_map)
    points = create_point_cloud(disp_map, im1_rgb)

    plt.imshow(disp_map, cmap='gray'), plt.show()

    plot_point_cloud(points)


    # ALTERNATIVE, CV2 Implementation
    # Initialize the stereo block matching object 
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # # Compute the disparity image
    # disparity = stereo.compute(cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
    # min = disparity.min()
    # max = disparity.max()
    # disparity = np.uint8(6400 * (disparity - min) / (max - min))

    # plt.imshow(disparity, cmap='gray'), plt.show()



if __name__ == "__main__":
    main()
