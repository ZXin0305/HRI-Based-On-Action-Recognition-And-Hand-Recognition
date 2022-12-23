import shutil
import os
from IPython import embed
import random

def remove_file(old_path, new_path, test_num=1200):
    print(old_path)
    print(new_path)

    filelist = os.listdir(old_path)
    print(len(filelist))
    split_test_datalist = random.sample(filelist, test_num)

    work_num = 1
    for file in split_test_datalist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)

        shutil.move(src, dst)
        print(f"working {test_num}/{work_num}")
        work_num += 1
def remove_file_2(old_path, new_path1, new_path2):
    filelist = os.listdir(old_path)
    train_subject = ['s1','s3','s5','s7']
    test_subject = ['s2','s4','s6','s8']
    for file in filelist:
        subject_num = file.split('_')[1]
        src = os.path.join(old_path,file)
        if subject_num in train_subject:
            dst = os.path.join(new_path1,file)
        elif subject_num in test_subject:
            dst = os.path.join(new_path2,file)
        shutil.move(src, dst)
    
def remove_file_3(old_path, new_path1, new_path2):
    filelist = os.listdir(old_path)
    train_subject = ['s01','s02','s03','s04','s05']
    test_subject = ['s06','s07','s08','s09','s10']
    for file in filelist:
        subject_num = file.split('_')[1]
        src = os.path.join(old_path,file)
        if subject_num in train_subject:
            dst = os.path.join(new_path1,file)
        elif subject_num in test_subject:
            dst = os.path.join(new_path2,file)
        shutil.move(src, dst)

if __name__ == "__main__":
    # /media/xuchengjun/datasets/action_zx/4(previous)
    remove_file("/media/xuchengjun/datasets/action_zx/NEW/4", "/media/xuchengjun/datasets/action_zx/NEW/test_4")
    # remove_file_2("/media/xuchengjun/datasets/UTD-MAD/Skeleton","/media/xuchengjun/datasets/UTD-MAD/cs2/train","/media/xuchengjun/datasets/UTD-MAD/cs2/test")
    # remove_file_3("/media/xuchengjun/datasets/MSRAction3D/MSRAction3D_download/MSRAction3DSkeletonReal3D","/media/xuchengjun/datasets/MSRAction3D/cs2/train","/media/xuchengjun/datasets/MSRAction3D/cs2/test")