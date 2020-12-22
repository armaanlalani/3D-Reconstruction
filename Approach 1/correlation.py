import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import DISTANCE_THRESH, FOCAL_LENGTH, BASELINE, POINT_THRESHOLD, CORRELATION_WINDOW_SIZE, WINDOW_RANGE
from filters import gaussian_filter, post_smoothing


def filter_matches(matches):
    filtered_matches = []
    for u, v in matches:
        if u.distance < DISTANCE_THRESH * v.distance:
            filtered_matches.append(u)
    return filtered_matches


def get_src_and_dest(im1, im2):
    sift = cv2.xfeatures2d.SIFT_create()
    # Normalize and convert to 8 bit integers for sift
    # We only need this conversion if we smooth the images first
    kp1, desc1 = sift.detectAndCompute(cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), None)
    kp2, desc2 = sift.detectAndCompute(cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), None)

    # Match descriptors, find top 2 matches, then filter matches
    bf = cv2.BFMatcher()
    filtered_matches = filter_matches(bf.knnMatch(desc1, desc2, k=2))

    # im3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))
    # plt.imshow(im3)
    # plt.show()

    src, dest = [], []
    for u in filtered_matches:
        src.append(kp1[u.queryIdx].pt)
        dest.append(kp2[u.trainIdx].pt)    

    src = np.array(src, dtype=np.int32).reshape(-1, 1, 2)
    dest = np.array(dest, dtype=np.int32).reshape(-1, 1, 2)

    # im3 = cv2.drawMatches(im1, kp1, im2, kp2, filtered_matches, im2, flags=2)
    # plt.imshow(im3), plt.show()
    print('Number of point matches:', src.shape[0])
    return src, dest, filtered_matches, kp1, kp2


def post_process_disp_map(disp_map):
    # disp_map = gaussian_filter(disp_map)
    # disp_map = post_smoothing(disp_map, window_size=20)
    # disp_map = post_smoothing(disp_map, window_size=10)
    # disp_map = gaussian_filter(disp_map)
    # disp_map = post_smoothing(disp_map, window_size=5)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    # disp_map = post_smoothing(disp_map, window_size=4)
    return disp_map


def get_disp_map(ref_im, src, dest):
    """
    ref_im should have 3 channels (RGB)
    """
    h, w, _ = ref_im.shape
    disp_map = np.zeros((h, w))
    for src_pt, dest_pt in zip(src, dest):
        # Point order from SIFT is flipped
        x_r, y_r = src_pt
        x_t, y_t = dest_pt
        disparity = x_r - x_t
        disp_map[y_r, x_r] = disparity

    # plt.imshow(disp_map, cmap='gray')
    # plt.show()
    return disp_map


def get_window_matching_disp_map(im1, im2):
    h, w = im1.shape
    disp_map = np.zeros((h, w))
    i = 0
    while i < h - CORRELATION_WINDOW_SIZE:
        j = 0
        print('Correlation for row {}, {:.1f}% finished'.format(i, 100 * i / (h - CORRELATION_WINDOW_SIZE)))
        while j < w - CORRELATION_WINDOW_SIZE:
            window1 = im1[i:i + CORRELATION_WINDOW_SIZE, j:j + CORRELATION_WINDOW_SIZE].flatten()

            k = max(0, j - WINDOW_RANGE)
            min_abs_diff = float('inf')
            while k < j:
                window2 = im2[i:i + CORRELATION_WINDOW_SIZE, k:k + CORRELATION_WINDOW_SIZE].flatten()

                # Compute sum of abs differences over window
                abs_diff = np.sum(np.abs(np.subtract(window1, window2)))
                # Update minimum
                if abs_diff < min_abs_diff:
                    min_abs_diff = abs_diff

                k += 1
            
            # Update disparity with minimum found
            disp_map[i, j] = min_abs_diff
            j += 1
        i += 1

    return disp_map


def create_point_cloud(disp_map, ref_im):
    bf = BASELINE * FOCAL_LENGTH
    h, w = disp_map.shape[0], disp_map.shape[1]
    pts = np.zeros((h*w, 6))
    k = 0
    for x in range(h):
        for y in range(w):
            b, g, r = ref_im[x, y, :]
            disp = disp_map[x, y]
            
            if disp > POINT_THRESHOLD:
                # Method 1
                # z = bf / disp

                # Method 2
                z = np.multiply(disp_map[x, y], 6)

                pts[k, :] = np.array([x, y, z, r, g, b])
                k += 1
            # else:
            #     z = -100000
            #     pts[k, :] = np.array([x, y, z, r, g, b])
            #     k += 1
    
    return pts

            
