from stereoLib import *
from stereoConfig import *


def stereo(im1Path,im2Path,K1,K2,baseline):
    # reading images
    im1 = cv2.imread(im1Path)
    im2 = cv2.imread(im2Path)
    image_shape1 = im1.shape[:2]

    
    (height1, width1) = image_shape1
    pts1 = np.float32([[0,0],[0,width1-2],[0,height1-2],[width1-1,height1-1]])

    # # keypoint detection
    srcCoords,dstCoords,fMatches,keyPts1,keyPts2 = getS2D(im1,im2)
    src, dst = srcCoords.squeeze(), dstCoords.squeeze()
    # # generating homogenous points
    srchomo = genHomogenousPts(src)
    dsthomo = genHomogenousPts(dst)
    # # computing fundamental matrix
    F = getFundMat(src,dst)
    # # computing epipoles
    epiR,epiL = findEpiPoles(src,dst,F)

    # # computing essential matrix
    E = findEssential(F,K1,K2)
    # # obtaining extrinsics
    rt = findExtrinsics(dsthomo,E,K2)
    if rt != False:
        R,T = rt 

    # # generating rectifying homgraphies
    H1, H2 = getRectifHomogs(R,T,epiR,epiL)


    x_offset = 0.3
    y_offset = 0.5

    # # # premultiply homography matrices to account for offset by warping
    pre = np.array([[ 1 , 0 , y_offset],[ 0 , 1 , x_offset],[ 0 , 0 ,    1    ]],dtype=np.float32)



    # # H
    H1[2,2] = 1
    H2[2,2] = 1
    H1 = pre @ H1
    H2 = pre @ H2
    
    rect1 = rectify(im1,H1,focal,K1)
    rect2 = rectify(im2,H2,focal,K2)
    # Computing disparity
    windowSize = 3
    minDisp = 0
    numDisp = 64 - minDisp
    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,numDisparities=numDisp-minDisp, blockSize=windowSize,uniquenessRatio=2,speckleRange=15,speckleWindowSize=5,disp12MaxDiff=1,P1=8*3*(windowSize)**2,P2=32*3*(windowSize)**2)
    disp = stereo.compute(rect1,rect2).astype(np.float32)

    disp = ndimage.rotate(disp,90,reshape=False)

    flip = np.fliplr(disp)
    plt.imshow(flip)
    plt.show()
    depth = mydepth(flip,focal,baseline)
    plt.imshow(depth)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    stereo(im1Path='./motorcycle/im0.png',im2Path='./motorcycle/im1.png',K1=K1,K2=K2,baseline=baseline)







    


