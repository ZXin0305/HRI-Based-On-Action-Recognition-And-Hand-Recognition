import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

class CameraTopic(object):
    def __init__(self, topic_name, video_path):
        self.image = None
        self.color_convert = None
        self.cv_bridge = CvBridge()
        self.topic = topic_name
        self.image_sub = rospy.Subscriber(self.topic, Image, self.callback)
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 20.0,(1920, 1080))

    def callback(self, msg):
        # self.image = self.cv_bridge.imgmsg_to_cv2(msg)
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "rgba8")   #  bgr8  之前的
        self.color_convert = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
        if self.image is not None:

            self.out.write(self.color_convert)
            cv2.imshow('img', self.color_convert)
            cv2.waitKey(25)
        else:
            raise StopIteration

if __name__ == "__main__":
    rospy.init_node('Image_sub', anonymous=True)
    rospy.Rate(60).sleep()
    video_path = "/media/xuchengjun/disk/zx/videos/testhand.mp4"
    camera_topic = "/kinectSDK/color"    # /kinect2_2/hd/image_color  /kinectSDK/color
    cam = CameraTopic(camera_topic, video_path)
    rospy.spin()