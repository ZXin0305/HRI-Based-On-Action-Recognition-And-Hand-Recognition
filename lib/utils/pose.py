import cv2
import numpy as np


class Pose:
    num_kpts = 15
    sigmas = np.array([.79, .26, .79, .72, .62, 1.07, .87, .89, .79, .72, .62, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2    
    last_id = -1

    def __init__(self):
        super().__init__()
        pass
    
        
