"""
this is for generating COCO annotations 
"""

from pycocotools.coco import COCO
import numpy as np
import json
import os
from IPython import embed
from path import Path
root_dir = '/media/xuchengjun/datasets/coco_2017'
data_type = 'train2017'
anno_name = 'person_keypoints_{}.json'.format(data_type)
anno_file = os.path.join(root_dir, 'annotations_trainval2017', 'annotations', anno_name)
# output_json_file = os.path.join(root_dir, 'coco_keypoints_{}.json'.format(data_type))  # original
coco_kps = COCO(anno_file)

catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds)

COCO2CMUP = [-1, -1, -1, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]  # if do not use 'nose', COCO2CMUP[1] == -1

# def main():
#     output_json = dict()
#     output_json['root'] = []
#     count = 0
#     min_width = 1000
#     min_height = 1000
#     for i in range(len(imgIds)):
#         bodys = list()
#         img = coco_kps.loadImgs(imgIds[i])[0]
#         annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
#         annos = coco_kps.loadAnns(annIds)
#         for anno in annos:
#             if anno['num_keypoints'] < 3:
#                 continue
#             body = np.asarray(anno['keypoints'])
#             body.resize((17, 3))
#             body_new = np.zeros((15, 11))
#             for k in range(len(COCO2CMUP)):
#                 if COCO2CMUP[k] < 0:
#                     continue
#                 body_new[k][0] = body[COCO2CMUP[k]][0]
#                 body_new[k][1] = body[COCO2CMUP[k]][1]
#                 body_new[k][3] = body[COCO2CMUP[k]][2]  # 这个是标注的关节点的性质
#             middle_shoulder = (body[5] + body[6]) / 2
#             middle_hip = (body[11] + body[12]) / 2
#             # hip
#             body_new[2][0] = middle_hip[0]
#             body_new[2][1] = middle_hip[1]
#             body_new[2][3] = min(body[11][2], body[12][2])
#             # neck
#             # body_new[0][0] = (middle_shoulder[0] - middle_hip[0])*0.185 + middle_shoulder[0]
#             # body_new[0][1] = (middle_shoulder[1] - middle_hip[1])*0.185 + middle_shoulder[1]
#             body_new[0][0] = middle_shoulder[0]
#             body_new[0][1] = middle_shoulder[1]
#             body_new[0][3] = min(body_new[2][3], body[5][2], body[6][2])

#             #head top (using nose)
#             # body_new[1][0] = (body[0][0] - body_new[0][0]) + body[0][0]
#             # body_new[1][1] = (body[0][1] - body_new[0][1]) + body[0][1]
#             # body_new[1][3] = min(body[0][2], body_new[0][3])
#             body_new[1][0] = body[0][0]
#             body_new[1][1] = body[0][1]
#             body_new[1][3] = body[0][2]

#             body_new[:, 7] = img['width']  # fx
#             body_new[:, 8] = img['width']  # fy
#             body_new[:, 9] = img['width'] / 2  # cx
#             body_new[:, 10] = img['height'] / 2 # cy
#             bodys.append(body_new.tolist())
#         if len(bodys) < 1:
#             continue
#         this_pic = dict()
#         this_pic["dataset"] = "COCO"
#         # this_pic["img_paths"] = data_type + "/" + img['file_name']
#         this_pic["img_paths"] = 'train2017' + "/" + img['file_name']
#         this_pic["img_width"] = img['width']
#         this_pic["img_height"] = img['height']
#         this_pic["image_id"] = img['id']
#         this_pic["cam_id"] = 0
#         this_pic["bodys"] = bodys
#         this_pic["isValidation"] = 0  # 0 for train, 1 for test
#         output_json["root"].append(this_pic)
#         count += 1
#         min_width = min(min_width, img['width'])
#         min_height = min(min_height, img['height'])
#         print(f"writed {img['file_name']}, total: {count}")

#     with open(output_json_file, 'w') as f:
#         json.dump(output_json, f)
#     print("Generated {} annotations, min width is {}, min height is {}.".format(count, min_width, min_height))

def main():
    output_json = dict()
    count = 0
    non_useful = 0
    min_width = 1000
    min_height = 1000
    output_root_path = Path('/media/xuchengjun/datasets/COCO')
    for i in range(len(imgIds)):
        output_json_file = output_root_path / f'{imgIds[i]}.json'
        bodys = list()
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
        annos = coco_kps.loadAnns(annIds)
        for anno in annos:
            if anno['num_keypoints'] < 3:
                continue
            body = np.asarray(anno['keypoints'])
            body.resize((17, 3))  #开始具有17个关节点
            body_new = np.zeros((15, 11))  #新的姿态中具有15个关节点，并且里面放有11中数据
            for k in range(len(COCO2CMUP)):
                if COCO2CMUP[k] < 0:
                    continue
                body_new[k][0] = body[COCO2CMUP[k]][0]
                body_new[k][1] = body[COCO2CMUP[k]][1]
                body_new[k][3] = body[COCO2CMUP[k]][2]  # 这个是标注的关节点的性质
            middle_shoulder = (body[5] + body[6]) / 2
            middle_hip = (body[11] + body[12]) / 2
            # hip
            body_new[2][0] = middle_hip[0]
            body_new[2][1] = middle_hip[1]
            body_new[2][3] = min(body[11][2], body[12][2])
            # neck
            # body_new[0][0] = (middle_shoulder[0] - middle_hip[0])*0.185 + middle_shoulder[0]
            # body_new[0][1] = (middle_shoulder[1] - middle_hip[1])*0.185 + middle_shoulder[1]
            body_new[0][0] = middle_shoulder[0]
            body_new[0][1] = middle_shoulder[1]
            body_new[0][3] = min(body_new[2][3], body[5][2], body[6][2])

            #head top (using nose)
            # body_new[1][0] = (body[0][0] - body_new[0][0]) + body[0][0]
            # body_new[1][1] = (body[0][1] - body_new[0][1]) + body[0][1]
            # body_new[1][3] = min(body[0][2], body_new[0][3])
            #(using nose)
            body_new[1][0] = body[0][0]
            body_new[1][1] = body[0][1]
            body_new[1][3] = body[0][2]

            body_new[:, 7] = img['width']  # fx
            body_new[:, 8] = img['width']  # fy
            body_new[:, 9] = img['width'] / 2  # cx
            body_new[:, 10] = img['height'] / 2 # cy
            bodys.append(body_new.tolist())
        if len(bodys) < 1:
            non_useful += 1
            continue
        output_json = dict()
        output_json["dataset"] = "COCO"
        output_json["img_paths"] = 'train2017' + "/" + img['file_name']
        output_json["img_width"] = img['width']
        output_json["img_height"] = img['height']
        output_json["image_id"] = img['id']
        output_json["cam_id"] = 0
        output_json["bodys"] = bodys
        output_json["isValidation"] = 0  # 0 for train, 1 for test
        count += 1
        min_width = min(min_width, img['width'])
        min_height = min(min_height, img['height'])

        with open(output_json_file, 'w') as f:
            json.dump(output_json, f)

    print("Generated {} annotations, min width is {}, min height is {}.".format(count, min_width, min_height))

if __name__ == "__main__":
    main()


