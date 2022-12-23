import random

sample_len = 1000
class GetDataset():
    def __init__(self, sub_dirs, useful_train_dirs, useful_img_dirs_train, \
                 useful_val_dirs, useful_img_dirs_val):
        self.sub_dirs = sub_dirs
        self.useful_train_dirs = useful_train_dirs
        self.useful_img_dirs_train = useful_img_dirs_train
        self.useful_val_dirs = useful_val_dirs
        self.useful_img_dirs_val = useful_img_dirs_val

    def get_train_data(self):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_train_dirs:  # 最终用到的训练集有四种
                # print(f'{sub_dir.basename()}')
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  # annotations 这里没有文件夹，是所有的
                # sample_annotation_files = random.sample(annotation_files, 6000)

                for img_dir in img_dirs:
                    if img_dir.basename() in self.useful_img_dirs_train:
                        # imgs = img_dir.files()  #len(imgs) == 16716 所有的数据集
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json')  # 标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):  # 读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  # 只要这个标签的文件数值就好
                                img_path = img_dir.split('panoptic-toolbox/')[-1] / (img_dir.basename() +  '_' + anno_num + '.jpg')
                                data_list.append((img_path, annotation_files[idx], cali_file_path,
                                                  img_dir.basename()))  # img_dir.basename()　--> 主要是为了得到对应的相机参数
        # if len(data_list) >= sample_len:
        #     print('sample ..')
        #     # random.shuffle(data_list)
        #     # data_list = random.sample(data_list,sample_len)
        # else:
        #     print(f'{useful_train_dirs}: {len(data_list)} ')

        random.shuffle(data_list)
        data_list = data_list[0:160000]  #16万张图片
        print(f'{useful_train_dirs}: {len(data_list)} ')
        return data_list

    def get_val_data(self):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_val_dirs:
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  # annotations 这里没有文件夹，是所有的

                for img_dir in img_dirs:
                    if img_dir.basename() in self.useful_img_dirs_val:
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json')  # 标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):  # 读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  # 只要这个标签的文件数值就好
                                img_path = img_dir.split('panoptic-toolbox/')[-1] / (img_dir.basename() +  '_' + anno_num + '.jpg')

                                data_list.append((img_path, annotation_files[idx], cali_file_path,
                                                  img_dir.basename()))  # img_dir.basename()　--> 主要是为了得到对应的相机参数
        random.shuffle(data_list)
        data_list = data_list[0:2000]  #每个类别就2000
        print(f'{useful_train_dirs}: {len(data_list)} ')
        return data_list


if __name__ == '__main__':

    import sys
    sys.path.append('/home/xuchengjun/ZXin/smap')
    from path import Path
    from IPython import embed
    from lib.utils.tools import read_json
    from lib.preprocess.project import reproject
    import json
    import numpy as np
    import os
    import time

    dataset_path = Path('/media/xuchengjun/disk/datasets/panoptic-toolbox')
    sub_dirs = dataset_path.dirs()
    useful_train_dirs = ['160906_pizza1', '160422_ultimatum1', '170407_haggling_a1']  # '170221_haggling_b1', '160906_pizza1','160422_ultimatum1' 161029_sports1
    useful_val_dirs = ['160906_pizza1']  # '170407_haggling_a1' 没有00_16, 00_30的相机内参
    useful_img_dirs_train = ['00_00','00_01','00_02','00_03','00_04','00_05','00_06','00_07','00_08','00_09']  # ,'00_01','00_02','00_03','00_04','00_05','00_06','00_07','00_08','00_09',
    useful_img_dirs_val = ['00_16','00_30']
    # print(useful_img_dirs_train)
    print(useful_val_dirs, '\t', useful_img_dirs_val)

    get_data = GetDataset(sub_dirs, useful_train_dirs, useful_img_dirs_train, useful_val_dirs, useful_img_dirs_val)
    # train_data_list = get_data.get_train_data()
    train_data_list = get_data.get_val_data()

    """
    total 29 videos --> useful 426099 imgs & non-useful 41950 imgs 但是smap原文中也只用了160k张图片
    """

    #这个是分开成单独的文件放在一起 
    min_width = 1000
    min_height = 1000
    human = 0
    no_human = 0
    s_time = time.time()
    #  /media/xuchengjun/datasets/CMU/160422_ultimatum1
    #  /media/xuchengjun/datasets/CMU/train
    output_root_path = Path('/media/xuchengjun/disk/datasets/CMU/160906_pizza1/val')  #处理后的数据存放位置
    if not output_root_path.exists():
        os.makedirs(output_root_path)
        print(f'creating gt json path --> {output_root_path}')
    for idx in range(len(train_data_list)):
        img_path , anno_path = train_data_list[idx][0] , train_data_list[idx][1]
        cali_path , cam_id = train_data_list[idx][2] , train_data_list[idx][3]
        
        anno_file = read_json(anno_path)
        cali_file = read_json(cali_path)
        if anno_file == None:
            print(train_data_list[idx])
        cam_id = str(cam_id)

        lnum , rnum = int(cam_id.split('_')[0]) , int(cam_id.split('_')[1])

        cam_coors , pixel_coors , skel_with_conf , cam, resolution = reproject(anno_file,cali_file,(lnum,rnum))
        if len(cam_coors) < 1:
            no_human += 1
            continue

        tmp = str(img_path).split('/')
        img_anno_name = tmp[-4] + '--' + tmp[-2] + "--" + tmp[-1].split('.')[0].split('_')[-1] 

        # output_json_root = dataset_path / f'{tmp[-4]}' / 'json_file'          #/media/xuchengjun/datasets/CMU/170407_haggling_a1/json_file

        # json_sub_dirs = output_json_root / f'{tmp[-2]}'

        # if not json_sub_dirs.exists():
        #     os.makedirs(json_sub_dirs)

        # output_json_path = json_sub_dirs / f'{img_anno_name}.json'
        output_json_path = output_root_path / f'{img_anno_name}.json'
        output_json = dict()

        bodys = list()
        for i in range(len(cam_coors)):
            body_new = np.zeros((15,11))

            for jtype in range(15):  # 刚好前15个
                body_new[jtype][0] = pixel_coors[i][0][jtype]   # x  (pixel)
                body_new[jtype][1] = pixel_coors[i][1][jtype]   # y  (pixel)
                body_new[jtype][2] = pixel_coors[i][2][jtype]   # Z  (cam)
                # if skel_with_conf[i][3][jtype] >= 0.2:
                body_new[jtype][3] = 2
                body_new[jtype][4] = cam_coors[i][0][jtype]     # X   (cam)
                body_new[jtype][5] = cam_coors[i][1][jtype]     # Y   (cam)
                body_new[jtype][6] = cam_coors[i][2][jtype]     # Z   (cam)
                body_new[jtype][7] = cam[0, 0]  # fx
                body_new[jtype][8] = cam[1, 1]  # fy
                body_new[jtype][9] = cam[0, 2]  # cx
                body_new[jtype][10] = cam[1, 2]  # cy
            bodys.append(body_new.tolist())

        output_json['dataset'] = 'CMU'
        output_json['img_paths'] = img_path.split('toolbox/')[-1]
        output_json['img_width'] = resolution[0]
        output_json['img_height'] = resolution[1]
        output_json['image_id'] = img_path.split('/')[-1].split('.')[-2]
        output_json['cam_id'] = cam_id
        output_json['bodys'] = bodys
        output_json["isValidation"] = 1    
    
        min_width = min(min_width, resolution[0])
        min_height = min(min_height, resolution[1])
        
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f)
        print('working .. {} / {}'.format(human, len(train_data_list)))
        
        # if human > 2000:
        #     break
        
        human += 1

    e_time = time.time()
    print(f'min_width: {min_width} \t min_height: {min_height}')
    print(f'done .. total_useful: {human}, no_human: {no_human}')
    print(f'using time --> {(e_time - s_time) / 3600}')
    
    
    # -------------------------------------------------------------------------------------------------------
    # 这里是把所有的都保存在一个文件中
    # output_json_file = Path('/media/xuchengjun/datasets/CMU/CMU.json')
    # count = 1
    # no_human = 0
    # s_time = time.time()
    # output_json = dict()
    # output_json['root'] = []
    # min_width = 1000
    # min_height = 1000
    # for idx in range(len(train_data_list)):
    #     img_path, anno_path = train_data_list[idx][0], train_data_list[idx][1]
    #     cali_path, cam_id = train_data_list[idx][2], train_data_list[idx][3]
    #     anno_file = read_json(anno_path)
    #     cali_file = read_json(cali_path)
    #     cam_id_str = str(cam_id)
    #     lnum, rnum = int(cam_id_str.split('_')[0]), int(cam_id_str.split('_')[1])
    #     # cam_coors, pixel_coors --> list:[array, array, ...]
    #     cam_coors, pixel_coors, skel_with_conf, cam, resolution = reproject(anno_file, cali_file, (lnum, rnum))

    #     if len(cam_coors) < 1:  # not include human
    #         no_human += 1
    #         continue

    #     bodys = list()
    #     for i in range(len(cam_coors)):
    #         body_new = np.zeros((15,11))

    #         for jtype in range(15):  # 刚好前15个
    #             body_new[jtype][0] = pixel_coors[i][0][jtype]   # x  (pixel)
    #             body_new[jtype][1] = pixel_coors[i][1][jtype]   # y  (pixel)
    #             body_new[jtype][2] = pixel_coors[i][2][jtype]   # Z  (cam)
    #             # if skel_with_conf[i][3][jtype] >= 0.2:
    #             body_new[jtype][3] = 2
    #             body_new[jtype][4] = cam_coors[i][0][jtype]     # X   (cam)
    #             body_new[jtype][5] = cam_coors[i][1][jtype]     # Y   (cam)
    #             body_new[jtype][6] = cam_coors[i][2][jtype]     # Z   (cam)
    #             body_new[jtype][7] = cam[0, 0]  # fx
    #             body_new[jtype][8] = cam[1, 1]  # fy
    #             body_new[jtype][9] = cam[0, 2]  # cx
    #             body_new[jtype][10] = cam[1, 2]  # cy
    #         bodys.append(body_new.tolist())

    #     this_pic = dict()
    #     this_pic['dataset'] = 'CMU'
    #     this_pic['img_paths'] = img_path
    #     this_pic['img_width'] = resolution[0]
    #     this_pic['img_height'] = resolution[1]
    #     this_pic['image_id'] = img_path.split('/')[-1].split('.')[-2]
    #     this_pic['cam_id'] = cam_id
    #     this_pic['bodys'] = bodys
    #     this_pic["isValidation"] = 0    
    #     output_json["root"].append(this_pic)

    #     min_width = min(min_width, resolution[0])
    #     min_height = min(min_height, resolution[1])

    #     print(f'working .. {count} / {len(train_data_list)}')
    #     count += 1

    # with open(output_json_file, 'w') as f:
    #     json.dump(output_json, f)
    # e_time = time.time()
    # print(f'min_width: {min_width} \t min_height: {min_height}')
    # print(f'done .. total_useful: {count}, no_human: {no_human}')
    # ------------------------------------------------------------------------------------------------------------------

