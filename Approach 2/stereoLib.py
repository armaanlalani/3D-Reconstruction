import numpy as np
import cv2 
from PIL import Image 
import matplotlib.pyplot as plt 
from scipy import ndimage
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D


def filterMatches(matches,thresholdRatio):
    """
    Filter matches based on threshold value
    """
    filtered = []
    for (m1,m2) in matches:
        if (m1.distance / m2.distance) < thresholdRatio:
            filtered += [m1]
    return filtered
    

def getS2D(im1,im2,thresholdRatio=0.75):
    """
        Obtain best corresponding points between left and right images
    """
    sift = cv2.xfeatures2d.SIFT_create()

    keyPts1, desc1 = sift.detectAndCompute(im1,None)
    keyPts2, desc2 = sift.detectAndCompute(im2,None)

    # get BF matcher obj 
    bfm = cv2.BFMatcher()
    matches = bfm.knnMatch(desc1,desc2,k=2)

    fMatches = filterMatches(matches,thresholdRatio=thresholdRatio)
    
    # print(len(fMatches),len(keyPts1),len(keyPts2))
    srcCoords = [keyPts1[m.queryIdx].pt for m in fMatches]
    dstCoords = [keyPts2[m.queryIdx].pt for m in fMatches]

    # reshaping accordingly to notation of cv and matplotlib
    srcCoords = (np.array(srcCoords,dtype=np.float32)).reshape(-1,1,2)
    dstCoords = (np.array(dstCoords,dtype=np.float32)).reshape(-1,1,2)

    return srcCoords,dstCoords,fMatches,keyPts1,keyPts2


def normalizeTransformation(srcCoords,dstCoords):
    """
        Provide normalizing transforms for both left and right images (corresponsing points used)

        Normalize points such that their centroid is mapped to the origin of the 
        camera's coordinate system. RMS distance is from each point to said origin is sqrt(2)

        Return:
            Tl --> Normalizing transform for left image
            Tr --> Normalizing transformation for right images

    """
    # print(max(srcCoords[:,0]),max(srcCoords[:,1]))
    numPts = srcCoords.shape[0]

    lcent = np.zeros((srcCoords.shape))
    lu = srcCoords[:,0].mean()
    lv = srcCoords[:,1].mean()

    rcent = np.zeros((dstCoords.shape))
    ru = dstCoords[:,0].mean()
    rv = dstCoords[:,1].mean()

    lcent[:,0] = srcCoords[:,0] - lu
    lcent[:,1] = srcCoords[:,1] - lv

    rcent[:,0] = dstCoords[:,0] - ru
    rcent[:,1] = dstCoords[:,1] - rv

    sl = np.sqrt(2)
    sr = np.sqrt(2)

    sl /= np.sqrt(
        ((lcent[:,0]**2) + (lcent[:,1]**2)).mean()
    )

    sr /= np.sqrt(
        ((rcent[:,0]**2) + (rcent[:,1]**2)).mean()
    )

    Tl = np.matmul(
        np.array([[sl,0,0],[0,sl,0],[0,0,1]]),
        np.array([[1,0,-lu],[0,1,-lv],[0,0,1]])
    )
    Tr = np.matmul(
        np.array([[sr,0,0],[0,sr,0],[0,0,1]]),
        np.array([[1,0,-ru],[0,1,-rv],[0,0,1]])
    )

    return Tl,Tr


def getFundMat(srcCoords,dstCoords):

    """
        Compute fundamental matrix given corresponding points between two images
        using the Normalized 8-point algorithm (used number src.shape[0] points)

        srcCoords --> Corresponding points for left image
        dstCoords --> Corresponding points for right image

        Return:
            F --> Fundamental Matrix
        
    """

    Tl,Tr = normalizeTransformation(srcCoords,dstCoords)

    lnorm = np.hstack((srcCoords,np.ones((srcCoords.shape[0],1))))
    rnorm = np.hstack((dstCoords,np.ones((dstCoords.shape[0],1))))

    lnorm = np.matmul(Tl,lnorm.T)
    rnorm = np.matmul(Tr,rnorm.T)

    A = np.zeros((srcCoords.shape[0],9))

    for i in range(srcCoords.shape[0]):
        ptl = lnorm[:,i]
        ptr = rnorm[:,i]

        row = [ptr[0]*ptl[0],ptr[0]*ptl[1],ptr[0],ptr[1]*ptl[0],ptr[1]*ptl[1],ptr[1],ptl[0],ptl[1],1]
        A[i,:] = np.array(row)

    
    u,s,v = np.linalg.svd(A)

    idx = (np.where(s == min(s)))[0][0]
    # print(idx[0][0])
    f = v[idx,:]
    F = f.reshape((3,3))
    
    uf,sf,vf = np.linalg.svd(F)

    sf = np.where( (sf != min(sf)),sf,0)
    sf = np.diag(sf)
    Fp = (uf @ sf) @ vf
    
    FOrig = (Tr.T @ Fp) @ Tl
    # print(FOrig)
 
    
    return FOrig

def vec2Cross(a):
    """
        Compute Matrix cross product representation of vector, skew:
            if a is length n, skew is nxn such that:
                (skew)(b) = a x b 

    """
    skew = np.array([
        [0,-a[2,0],a[1,0]],[a[2,0],0,-a[0,0]], [-a[2,0],a[0,0],0]
    ])
    return skew

def findEpiPoles(src,dst,F):
    """
        Compute the left and right epipoles such that the left epipole is in the right
        null space of F and the right epipole is in the left null space of F

        Input:
            src --> corresponding points in left image
            dst --> corresponding points in right image

        Return:
            epiL --> Left epipole
            epiR --> Right epipole
        
    """


    uf, sf, vf = np.linalg.svd(F)
    idx = (np.where(sf == min(sf)))[0][0]
    epiL = (vf[idx,:]).reshape((3,1))    
    epiR = (uf[:,idx]).reshape((3,1))
    
    return epiR,epiL


def findEssential(F,K1,K2):
    """
        Compute the Essential matrix, E, given the Fundamental Matrix and Intrinsic Parameters such that:

        E = (K2.T)F(K1)

        Input: 
            F --> Fundamental Matrix
            K1 --> Camera intrinsics of left camera
            K2 --> Camera intrinsics of right camera
    
        Return:

            E --> Essential Matric
    """

    # k2 is right and k1 is left
    # inverse transpose same thing as transpose inverse
    tmp = np.linalg.inv(K2.T)
    E = (np.linalg.inv(tmp)) @ F
    E = (E @ K1)
    
    u, s, vt = np.linalg.svd(E)
    if s[-1] != 0:
        s[-1] = 0

    s /= max(s)
    E = (u @ np.diag(s)) @ vt
    
    return E
            
def isInFront(dst,R,t,K2):
    """
    Determine whether or not 2 dimensional points get mapped to points in front of cameras

    Input: 
        dst --> corresponding points in right camera system
        R --> Rotation matrix (3x3)
        t --> Translation vector (3x1)
        K2 --> Intrinsic Parameters of right Camera

    Return:
        Bool: True if in front, False o/w


    """

    normdst = (np.linalg.inv(K2) @ dst.T).T
    # print(dst.shape)
    Rot1 = np.block([
        [R,np.zeros((3,1))],
        [np.zeros((1,3)),1]
    ])
    
    t1 = np.block([
            [np.eye(3),t],
            [np.zeros((1,3)),1],
        ])
    
    
    store = np.hstack((normdst,np.ones((normdst.shape[0],1))))
    rtinv = np.linalg.inv((Rot1 @ t1))

    store3d = rtinv @ (store.T)
    check = np.where(store3d[2,:] < 0)[0]
    if check.size == 0:
        return False

    return True


def findExtrinsics(dst,E,K2):
    """
        Extract correct extrinsic parameters such that points are infront of both cameras

        Input:
            dst --> Corresponding points in right Image
            E --> Essential Matrix
            K2 --> Right camera extrinsics

        Return:

            R --> Rotation Matrix (3x3)
            t --> Normalized translation vector (3x1)
    """
    # src and dst must be homogenous pts 
    R1,R2, t = cv2.decomposeEssentialMat(E)
    R = R1
    T = t
    if not isInFront(dst,R,T,K2):
        T = -t
        if not isInFront(dst,R,T,K2):
            R = R2
            if not isInFront(dst,R,T,K2):
                T = t
                if not isInFront(dst,R,T,K2):
                    return False    
    return R,T

def genHomogenousPts(pts):
    """
        Given a set of points, return the corresponding homogenous points
    """
    numPts = pts.shape[0]
    homogen = np.hstack((pts,np.ones((numPts,1))))
    return homogen
    

def getRectifHomogs(R,T,epiR,epiL):

    """
    Compute the homographies need to rectify the two images

    Input:\n
        \tR --> Rotation Matrix
        \tT --> Normalized translation vector
        \tepiR --> Right epipole
        \tepiL --> Left epipole
        
    Return:\n
        \tH1 --> Homography to rectify first image
        \tH2 --> Homography ro rectify second image
    """




    r1 = T.reshape((3,1))
    r2 = (np.array([-T[1],T[0],0]) / np.sqrt((T[0]**2) + (T[1]**2))).reshape((3,1))

    r3 = np.cross(r1,r2,axis=0)
    
    Rrect = -(np.array([r1.T,r2.T,r3.T]).astype(float)).squeeze()
    
    H1 = Rrect
    H2 = (R @ Rrect).astype(float)
    
    return H1,H2

def rectify(im,H,f,K):

    """
        Rectify an image. 

        Input:
            im --> Array representation of image
            H --> Rectifying transform
            f --> focal length
            K --> intrinsics

        Return:
            rectified --> Return a rectified image in pixel coordinates
    """
    
    height, width = im.shape[:2]
    # print(H)
    idxy, idxx = np.indices((height,width),dtype=np.float32)

    homoIdxs = np.array([idxx.ravel(),idxy.ravel(),(np.full_like(idxx,fill_value=f)).ravel()])
    rect = H @ homoIdxs
    
    rect /= rect[-1]
    rect = K @ rect
    mapX, mapY = rect[:-1] / rect[-1]

    mapX = mapX.reshape((height,width)).astype(np.float32)
    mapY = mapY.reshape((height,width)).astype(np.float32)


    mapX = mapX.reshape((height,width)).astype(np.float32)
    mapY = mapY.reshape((height,width)).astype(np.float32)

    rectified = cv2.remap(im,mapX,mapY,cv2.INTER_LINEAR)
    
    return rectified



def mydepth(arr,focal,baseline):
    """
        Compute depth provided disparity values

        Input:
            arr --> Disparity matrix in pixel coords
            focal --> focal length
            baseline --> Baseline distance b/w cameras

        Return:
            depth --> depth for corresponding pixels
    """
    num = (focal) * (baseline/100)

    denomFlip = np.reciprocal(arr + 0.001)

    depth = num * denomFlip

    return depth
