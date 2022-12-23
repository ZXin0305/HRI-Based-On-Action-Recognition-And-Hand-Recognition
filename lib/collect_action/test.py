import cv2
import os
from IPython import embed
import numpy as np


if __name__ == "__main__":

    dir_num = 0
    dir_path = "/media/xuchengjun/zx/process/03"
    image_dir_root = "/media/xuchengjun/zx/process/4"

    image_dirs = os.listdir(image_dir_root)

    for index, image_dir in enumerate(image_dirs):
        if index % 10 == 0:
            image_dir_path = os.path.join(image_dir_root, image_dir)
            image_list = os.listdir(image_dir_path)
            image_idx_list_ori = [int(image_name.split(".")[0]) for image_name in image_list]
            sort_list = np.array(image_idx_list_ori).argsort()
            image_idx_list = list(np.array(image_idx_list_ori)[sort_list])

            for idx in range(len(image_idx_list)):  # --> int
                if idx % 2 == 0:
                    dir_name = os.path.join(dir_path, str(dir_num))  # 存放新的image
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    image_idx = image_idx_list[idx]
                    image_path = os.path.join(image_dir_path, str(image_idx) + '.jpg')

                    image = cv2.imread(image_path)
                    for i in range(10):
                        new_image_name = os.path.join(dir_name, str(i) + '.jpg')
                        cv2.imwrite(new_image_name, image)

                    dir_num += 1