import cv2
import numpy as np
import glob
from PIL import Image
from PIL import ExifTags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import plotly.io as pio

pio.renderers.default = 'browser'

def filter():
    path = glob.glob('./Calibration/*') # gets all the checkerboard in the calibration folder
    for filename in path:
        image = Image.open(filename)
        image = image.convert(mode='L')
        data = np.asarray(image)
        data = np.where(data>50,255,data) # filters out all pixels that are not white or black and changes them to be white
        image = Image.fromarray(data)
        image.save(filename) # saves the new file

def calibrate():
    size = (9,6) # number of inner corners in the checkerboard used
    obj_pts, img_pts = [], []
    
    objp = np.zeros((np.prod(size),3),dtype=np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

    path = glob.glob('./Calibration/*')

    for filename in path: # iterates through the images in calibration
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray,size,None) # attempts to locate the chessboard

        if ret:
            print('Found')
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            obj_pts.append(objp) # adds the object points if found
            img_pts.append(corners) # adds the image points if found
            
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts,img_pts,gray.shape[::-1],None,None) # determines the camera intrinsics based on the object and image points
    np.save("./ret", ret) # saves the numpy arrays to files for future use
    np.save("./K", K)
    np.save("./dist", dist)
    np.save("./rvecs", rvecs)
    np.save("./tvecs", tvecs)

def focal_length():
    path = glob.glob('./Calibration/*.jpg')
    exif_img = Image.open(path[0]) # opens the first image in the calibration folder
    exif_data = {ExifTags.TAGS[k]:v for k, v in exif_img._getexif().items() if k in ExifTags.TAGS} # determines the focal length using PIL
    focal_length = exif_data['FocalLength']
    return focal_length

def gaussian(x, stddev):
    return 1 / np.sqrt(2 * np.pi) * stddev * np.exp(-(x ** 2) / (2 * stddev ** 2)) # gaussian function

def compute_kernel(sigma):
    dim = 2 * int(2 * sigma + 0.5) + 1 # determines the size of the kernel using sigma --> ensures the size of the kernel is odd
    k_gaussian = np.linspace(-(dim//2), dim//2, dim) # creates the array that will be used to create the outer product
    for i in range(len(k_gaussian)):
        k_gaussian[i] = gaussian(k_gaussian[i], sigma) # 1D gaussian function
    k_gaussian = np.outer(k_gaussian.T, k_gaussian.T) # creates a 2D gaussian function by taking the outer product of the two 1D gaussian functions
    k_gaussian = k_gaussian / k_gaussian.max() # normalizes the kernel to ensure the maximum value is 1
    return k_gaussian

def convolve(im, kern):
    im_h = im.shape[0]
    im_w = im.shape[1]
    kern_h = kern.shape[0]
    kern_w = kern.shape[1]
    output = np.zeros((im_h, im_w))

    kern_size = kern_h * kern_w

    add = [int((kern_h-1)/2), int((kern_w-1)/2)] # size of the additional padded height and weight when filter is placed at edges of the image
    new_im = np.zeros((im_h + 2 * add[0], im_w + 2 * add[1])) # dimensions of the padded image
    new_im[add[0] : im_h + add[0], add[1] : im_w + add[1]] = im # sets the non-padded pixels of the padded image to the pixels of the image being convolved

    for i in range(im_h):
        for j in range(im_w):
            result = kern * new_im[i : i + kern_h, j : j + kern_w] # elementwise multiplication of kernel and appropriate pixels
            output[i, j] = np.sum(result) # adds the elements of the elementwise multiplication
    output = output / kern_size # reduction of pixel values based on kernel size

    return output

def filtered(match):
    filtered = []
    for i, j in match:
        if 0.75 * j.distance > i.distance: # adds the reliable matches based on a chosen value of phi
            filtered.append(i)
    return filtered

def source_des(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None) # determines the key points and descriptors for image 1
    kp2, des2 = sift.detectAndCompute(image2,None) # determines the key points and descriptors for image 2

    brute_force = cv2.BFMatcher()
    matches = filtered(brute_force.knnMatch(des1, des2, k=2)) # determines the matches between the two images

    source = []
    descriptors = []
    for match in matches:
        source.append(kp1[match.queryIdx].pt)
        descriptors.append(kp2[match.trainIdx].pt) # adds the sources and descriptors to their respective arrays
    source = np.array(source, dtype=np.float32).reshape(-1,1,2)
    descriptors = np.array(descriptors, dtype=np.float32).reshape(-1,1,2)
    return source, descriptors, matches, kp1, kp2, des1, des2 

def findF(image1, image2):
    source, descriptors, matches, kp1, kp2, des1, des2 = source_des(image1, image2)
    F, mask = cv2.findFundamentalMat(source,descriptors,cv2.FM_LMEDS) # determines the fundamental matrix based on the matching points
    source = source[mask.ravel()==1]
    descriptors = descriptors[mask.ravel()==1]
    return F, source, descriptors

def rectify(image1, image2, K, F, d, source, descriptors):
    (h,w) = image1.shape[:2]
    size = (w,h)
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(source.ravel(),descriptors.ravel(),F,size) # determines homography matrices for rectification

    K_inv = np.linalg.inv(K)
    R1 = K_inv.dot(H1).dot(K)
    R2 = K_inv.dot(H2).dot(K) # determines rectification matrices based on homography matrix and camera intrinsice

    x1map, y1map = cv2.initUndistortRectifyMap(K,d,R1,K,size,cv2.CV_16SC2)
    x2map, y2map = cv2.initUndistortRectifyMap(K,d,R2,K,size,cv2.CV_16SC2) # determines the mapped points based on the recitifcation matrices

    rectify1 = cv2.remap(image1,x1map,y1map,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    rectify2 = cv2.remap(image2,x2map,y2map,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) # creates the rectified images

    return rectify1, rectify2

def depth_map(im1,im2,disparity,baseline,f):
    mask = disparity > disparity.min() # filters out disparities of 0
    depth = np.zeros(disparity.shape)
    depth = baseline * f / (disparity+0.0001) # determines the depth of each point using the disparity and focal length
    return mask,depth

def point_cloud_depth(im1,depth,mask,K):
    points = np.zeros((depth.shape[0]*depth.shape[1],3))
    color = np.zeros((depth.shape[0]*depth.shape[1],3))
    count = 0
    K_inv = np.linalg.inv(K)
    for i in range(depth.shape[0]): # iterates through the points of the image and adds their coordinates and rbg values
        for j in range(depth.shape[1]):
            if mask[i,j]:
                Q_im = np.array([i,j,1]) * (depth[i,j])
                Q_c = np.matmul(K_inv,Q_im)
                points[count,0] = i
                points[count,1] = j
                points[count,2] = depth[i,j]/1000
                color[count,:] = im1[i,j,:]
                count += 1
    points = points[:count,:]
    color = color[:count,:]
    result = np.zeros((points.shape[0],6))
    result[:,:3] = points
    result[:,3:] = color
    return result # returns the final matrix used for the point cloud

def plot_pointCloud(pc):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,1,0), (0,0,-1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0].flatten(),
        y=pc[:, 1].flatten(),
        z=-pc[:, 2].flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=pc[:, 3:][..., ::-1],
            opacity=0.8
        )
    )])
    fig.show()

if __name__ == '__main__':
    print(cv2.__version__)
    # filter()
    # calibrate()

    ret = np.load('./ret.npy')
    K = np.load('./K.npy')
    d = np.load('./dist.npy')
    R = np.load('./rvecs.npy')
    t = np.load('./tvecs.npy')
    f = int(focal_length())

    im1 = cv2.imread(os.path.join(os.getcwd(),'./Images/Right.jpg')) # gets the two images to be used
    im2 = cv2.imread(os.path.join(os.getcwd(),'./Images/Left.jpg'))
    
    F,source,descriptors = findF(im1,im2) # determines F and the matching points
    rectify1, rectify2 = rectify(im1,im2,K,F,d,source,descriptors) # gets the rectified images

    rect1_im = Image.fromarray(rectify1).convert(mode='L') # converts the rectified images to grayscale
    rect2_im = Image.fromarray(rectify2).convert(mode='L')

    rectify1_gray = np.array(rect1_im)
    rectify2_gray = np.array(rect2_im)

    window_size = 5
    min_disp = -1
    max_disp = 15
    num_disp = max_disp - min_disp
    
    stereo = cv2.StereoSGBM_create(minDisparity=-1,numDisparities=num_disp,blockSize=5, uniquenessRatio=5,speckleRange=5,speckleWindowSize=5,disp12MaxDiff=1)
    disparity = stereo.compute(rectify1_gray,rectify2_gray) # generates the disparity map
    plt.imshow(disparity,'gray')
    plt.show()
    
    minimum = np.min(disparity)
    disparity = disparity - minimum
    maximum = np.max(disparity)
    disparity = disparity / maximum
    disparity = disparity * 255 # ensures disparity is in a range of 0 to 255 (normalizes values)

    mask,depth = depth_map(rectify1,rectify2,disparity,10,f)
    depth = convolve(depth,compute_kernel(2))
    #plt.imshow(depth,'gray')
    #plt.show()
    result = point_cloud_depth(rectify1,depth,mask,K)
    plot_pointCloud(result)