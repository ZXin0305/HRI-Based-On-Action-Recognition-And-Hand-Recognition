from os import path as osp
import os
import json
import numpy as np
muco2mpi15 = [1, 0, 14, 5, 6, 7, 11, 12, 13, 2, 3, 4, 8, 9, 10]
def format_json(in_path='MuCo-3DHP.json', dataset='MuCo', out_dir=''):
    out_path = osp.join(out_dir, dataset)
    if not osp.exists(out_path):
        os.system("mkdir -p %s"%(out_path))
    with open(in_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    annot_dict = dict()
    for i, annot in enumerate(annotations):
        img_id = annot['image_id']
        if img_id not in annot_dict.keys():
            annot_dict[img_id] = []
        p2d = np.array(annot['keypoints_img'])[muco2mpi15, :]
        p3d = np.array(annot['keypoints_cam'])[muco2mpi15, :]
        vis = np.array(annot['keypoints_vis'])[muco2mpi15]
        pose = []
        for j in range(15):
            pose_info = [p2d[j, 0], p2d[j, 1], p3d[j, 2]/10, int(vis[j]*2), p3d[j, 0]/10, p3d[j, 1]/10, p3d[j, 2]/10]
            pose.append(pose_info)
        annot_dict[img_id].append(pose)

    output_json = dict()
    output_json['root'] = []
    for img_info in data['images']:
        cur_info = dict()
        cur_info['img_height'] = img_info['height']
        cur_info['img_width'] = img_info['width']
        cur_info['img_paths'] = osp.join('images', img_info['file_name'])
        cur_info['dataset'] = "MuCo"
        cur_info['isValidation'] = 0
        fx, fy = img_info['f']
        cx, cy = img_info['c']
        cur_info['bodys'] = annot_dict[img_info['id']]
        for nh in range(len(cur_info['bodys'])):
            for j in range(len(cur_info['bodys'][nh])):
                cur_info['bodys'][nh][j] += [fx, fy, cx, cy]
        output_json['root'].append(cur_info)
                
    with open(osp.join(out_path, '%s.json'%(dataset)), 'w') as f:
        json.dump(output_json, f)