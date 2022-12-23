export PROJECT_HOME='/home/xuchengjun/ZXin/smap'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python ori_test.py -p "/home/xuchengjun/ZXin/smap/pretrained/main_model.pth" \
-t generate_result \
-d test \
-rp "/home/xuchengjun/ZXin/smap/pretrained/refine.pth" \
--batch_size 16 \
--do_flip 1 \
--dataset_path "/media/xuchengjun/disk/datasets/mupots-3d-eval/MultiPersonTestSet"   # /path/to/custom/image_dir