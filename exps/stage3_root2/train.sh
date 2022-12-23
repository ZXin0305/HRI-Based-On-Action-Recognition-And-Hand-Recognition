export CUDA_VISIBLE_DEVICES="0,1,2"
export PROJECT_HOME='/home/xuchengjun/ZXin/smap'
export PYTHONPATH=$PYTHONPATH:$PROJECT_HOME
python config.py -log
python -m torch.distributed.launch --nproc_per_node=3 train.py