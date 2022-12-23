import cv2
import os

class VideoReaders:
    def __init__(self, file_name):
        self.file_name = file_name
        
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        # 3840 × 2160
        # 1920 x 1080
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, image = self.cap.read()
        if not was_read:
            raise StopIteration

        return image
    
if __name__ == '__main__':
    from IPython import  embed
    # /media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos
    # /home/xuchengjun/Videos
    # /media/xuchengjun/disk1/zx/Videos/20221108
    video_path = '/home/xuchengjun/Videos/coordinate.mp4'
    store_path = '/media/xuchengjun/disk1/zx/Videos/20221108/coordinate'

    if not os.path.exists(store_path):
        os.mkdir(store_path)
    frame_provider = VideoReaders(video_path)
    
    id = 0
    for image in frame_provider:
        image_path = os.path.join(store_path, str(id) + '.jpg')
        cv2.imwrite(image_path, image)
        print(f'working .. {id}')
        id += 1
    print('done ..', id)