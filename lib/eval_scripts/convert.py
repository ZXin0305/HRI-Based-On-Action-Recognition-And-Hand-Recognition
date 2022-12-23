import numpy as np
import scipy.io as scio
import json
import os
from IPython import embed

def convert(path=''):
    with open(path, 'r') as f:
        data = json.load(f)

    pairs_3d = data['3d_pairs']

    pose3d = dict()
    pose2d = dict()
    gt3d = dict()
    count = dict()
    for i in range(len(pairs_3d)):  
        name = pairs_3d[i]['image_path']
        name = name[name.index('TS'):]
        ts = int(name[(name.index('TS')+2):name.index('/')])
        if ts < 6:
            width, height = 2048, 2048
        elif ts <= 20:
            width, height = 1920, 1080
        else:
            raise NotImplementedError
        
        if ts not in count.keys():
            count[ts] = 1
        else:
            count[ts] += 1
        embed()
        pred_3ds = np.array(pairs_3d[i]['pred'])        
        gt_3ds = np.array(pairs_3d[i]['gt'])
        intri = gt_3ds[0,0,3:7]
        K = np.array([[intri[1], 0, intri[2]],
                      [0, intri[1], intri[3]],
                      [0, 0, 1]])

        pred_2ds = pairs_3d[i]['pred_2d'] 
        pred_2ds = np.array(pred_2ds )

        crop_x = 832
        crop_y = 512
        scale = min(crop_x / float(width), crop_y / float(height))
        adj_p2d = np.array([0, 0])

        if height * scale < crop_y:
            pad = (crop_y - height * scale) // 2
            adj_p2d = np.array([0, pad])
        if width * scale < crop_x:
            pad = (crop_x - width * scale) // 2
            adj_p2d = np.array([pad, 0])
        
        for ih in range(len(pred_2ds)):
            joint2d = np.array(pred_2ds[ih])
            joint2d = joint2d[:, :2]
            joint2d -= np.expand_dims(adj_p2d, axis=0)
            joint2d /= scale
            pred_2ds[ih, :, :2] = joint2d
        
        reproject = True
        if reproject:
            op_2ds = pred_2ds.copy()
            # deal with 3d (input: op_2ds, relZ)
            new_pred_3ds = pred_3ds.copy()
            iK = np.linalg.inv(K)
            for ih in range(new_pred_3ds.shape[0]):
                if ih > len(op_2ds)-1:
                    continue
                for ij in range(new_pred_3ds.shape[1]):
                    tmp2d = np.array([op_2ds[ih,ij][0], op_2ds[ih,ij][1], 1]).reshape([3,1])
                    new_pred_3ds[ih,ij,:3] = (new_pred_3ds[ih,ij,2]* iK @ tmp2d).squeeze()
                    if op_2ds[ih,ij][3] == 0:                  
                        new_pred_3ds[ih, ij, :] = pred_3ds[ih, ij, :]

        if reproject:
            pred_3ds = new_pred_3ds

        pose3d[name] = pred_3ds * 10 # nH x 15 x 3 
        pose3d[name][:, :, 3] /= 10
        pose2d[name] = pred_2ds # nH x 15 x 4
        gt3d[name] = gt_3ds * 10     # nH x 15 x 4        

    if True:
        root_path = "/media/xuchengjun/disk/datasets/mupots-3d-eval/result"
        pose3d_file_name = "pose3d.mat"
        pose2d_file_name = "pose2d.mat"
        mat3d = os.path.join(root_path, pose3d_file_name)
        mat2d = os.path.join(root_path, pose2d_file_name)
        scio.savemat(mat3d, {'preds_3d_kpt':pose3d})
        scio.savemat(mat2d, {'preds_2d_kpt':pose2d})

if __name__ == "__main__":
    convert('/media/xuchengjun/disk/datasets/mupots-3d-eval/result/stage3_root2_run_inference_test_.json')