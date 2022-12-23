import numpy as np

def reproject(anno_file, cali_file, cam_id):
    cam_coors = []
    pixel_coors = []
    skel_with_conf_list = []
    skel_with_conf = None
    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in cali_file['cameras']}
    # Convert data into numpy arrays for convenience --> all
    for k, cam in cameras.items():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3, 1))

    # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera
    cam = cameras[cam_id]  # 这个就是序号为:00_00的高清镜头的参数
    """
    Reproject 3D Body Keypoint onto the HD camera
    """

    for body in anno_file['bodies']:
        skel_with_conf = np.array(body['joints19']).reshape((-1, 4)).transpose()  # (19,4) --> (4,19) 最后一行是置信度
        skel_with_conf_list.append(skel_with_conf)
        cam_coor, pixel_coor = projectPoints(skel_with_conf[0:3, :], cam['K'], cam['R'], cam['t'], cam['distCoef'])
        cam_coors.append(cam_coor)
        pixel_coors.append(pixel_coor)

    return cam_coors, pixel_coors, skel_with_conf_list, cam['K'], cam['resolution']


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.array(R * X + t)  # x.shape --> (3,19) cam-coordinate
    cam_coors = np.array(x)

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    pixel_coors = x  # 其实这里的第三行还是在相机坐标系下的

    return cam_coors, pixel_coors