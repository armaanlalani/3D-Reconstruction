import os

from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go

from config import SCALE_FACTOR, DATA_PATH


def load_image(name, color=False):
    # NOTE: Image format (width, height), but Numpy array format becomes (height, width)
    print("Loading image...")
    full_name = name + '.png'
    im = np.array(resize_image(Image.open(os.path.join(DATA_PATH, full_name))))
    # Convert to grayscale for convolution
    if not color:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def resize_image(im):
    # Arg: Image object from pillow
    w = im.size[0]
    h = im.size[1]
    return im.resize((w // SCALE_FACTOR, h // SCALE_FACTOR))


def plot_point_cloud(pc):
    '''
    plots the Nx6 point cloud pc in 3D
    assumes (1,0,0), (0,1,0), (0,0,-1) as basis
    '''
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=-pc[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pc[:, 3:][..., ::-1],
            opacity=0.8
        )
    )])
    fig.show()


def reshape_pts(arr):
    reshaped = np.reshape(arr, (arr.shape[0], arr.shape[2]))
    return reshaped

