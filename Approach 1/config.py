import os

SCALE_FACTOR = 16
IM1 = 'im0'
IM2 = 'im1'
DATA_PATH = os.path.join(os.getcwd(), 'data/Motorcycle-perfect')
SIGMA = 0.4
FILTER_ITERATIONS = 1
# FILTER_ITERATIONS = 4
DISTANCE_THRESH = 0.764
BASELINE = 193.001
FOCAL_LENGTH = 3979.911
POST_WINDOW_SIZE = 4
SMOOTHING_THRESHOLD = 30
POINT_THRESHOLD = 0.05
CORRELATION_WINDOW_SIZE = 11
WINDOW_RANGE = 50