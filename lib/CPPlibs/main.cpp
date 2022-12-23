#include<iostream>
#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<ctime>
using namespace std;
using namespace cv;

const int resize_size = 160;
std::vector<cv::Point> res_points;  // 返回的边界框顶点
int global_frame = -1;               // 记录当前的帧数
cv::Mat global_rgb;
cv::Mat global_depth;

struct croppedResult{
    char* cropped_hand_rgb;
    char* cropped_hand_depth;
};

char* cropped_rgb_global = nullptr;
char* cropped_depth_global = nullptr;


//1. 只得到RGB图像
// cv::Mat depthimage, cv::Mat* cropped_hand, cv::Mat* cropped_hand_depth
char* draw_rectangle(double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image){
    res_points.clear();
    cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
    cv::Size size_of_initial_crop = cv::Size(rec_size * 2, rec_size * 2);  // 变成了两倍
    cv::Mat aligned_bounding_rect;
    cv::Mat Rotation_Matrix, bounding_rect_rotated;
    cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_32FC3);

    // 旋转框
    cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle); //返回旋转后的矩形信息,  角度是逆负顺正
    cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect(); // returns the minimal up-right integer rectangle containing the rotated rectangle,  这个返回的是旋转框的最小边界框
    cv::Size rotatedRectangle_size = rotatedRectangle.size;  // width and height of the rectangle after rotation, 旋转后的宽、高和之前的一样

    cv::Size diagonal_rectangle_size = diagonal_rectangle.size(); // size (width, height) of the rectangle, 旋转之后最小边界框的宽高
    cv::Point2f points[4];
    rotatedRectangle.points(points); // The points array for storing rectangle vertices. The order is bottomLeft, topLeft, topRight, bottomRight.
                                     // 旋转之后的四个顶点，顺时针

    cv::cvtColor(source_image, source_image, CV_BGRA2BGR);
    cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);  //提取出最小边界框中的图像 

    cv::Point aligned_bounding_rect_center;

    aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;  // 最小边界框的中心
    aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;

    // 保存旋转后的点,用另外一个函数返回
    if(res_points.empty()){
        res_points.emplace_back(points[0]);
        res_points.emplace_back(points[1]);
        res_points.emplace_back(points[2]);
        res_points.emplace_back(points[3]);
        // res_points.emplace_back(aligned_bounding_rect_center);
    }

    // 将提取出最小边界框进行旋转, 朝着反方向，因为要使图像变正， 所以这里的角度是 +90
    // 会有一部分显示不出来， 不过后面的cv::getRectSubPix() 会提取出中间的图像，刚好是旋转框旋转正之后的图像
    Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle+90, 1.0);
    cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);

    cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
    // cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);

    resized_rgb_crop = cv::Mat(cv::Size(resize_size, resize_size), CV_32FC3);
    cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);

    // cv::imwrite("./croped_img.jpg", resized_rgb_crop);

    // 将四个点进行连线
    // cv::line(input_image, points[0], points[1], cv::Scalar(0, 255, 0), 3);
    // cv::line(input_image, points[1], points[2], cv::Scalar(0, 255, 0), 3);
    // cv::line(input_image, points[2], points[3], cv::Scalar(0, 255, 0), 3);
    // cv::line(input_image, points[3], points[0], cv::Scalar(0, 255, 0), 3);
    // cv::imshow("RGB crop", rgb_cropped);
    // cv::imshow("source RGB", input_image);
    // cv::waitKey(0);

    if(!rgb_cropped.empty()){
        vector<uchar> data_encode;
        imencode(".jpg", resized_rgb_crop, data_encode);

        std::string str_encode(data_encode.begin(), data_encode.end());
        char* char_r = new char[str_encode.size() + 10];
        memcpy(char_r, str_encode.data(), sizeof(char)*(str_encode.size()));
        return char_r;
    }
    else{
        return nullptr;
    }
} 

//2.只得到深度图像
char* draw_rectangle_depth(double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image){
    cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
    cv::Size size_of_initial_crop = cv::Size(rec_size * 2, rec_size * 2);  // 变成了两倍
    cv::Mat aligned_bounding_rect;
    cv::Mat Rotation_Matrix, bounding_rect_rotated;
    cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_8UC1);

    // 旋转框
    cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle); //返回旋转后的矩形信息,  角度是逆负顺正
    cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect(); // returns the minimal up-right integer rectangle containing the rotated rectangle,  这个返回的是旋转框的最小边界框
    
    cv::Size rotatedRectangle_size = rotatedRectangle.size;  // width and height of the rectangle after rotation, 旋转后的宽、高和之前的一样
    cv::Size diagonal_rectangle_size = diagonal_rectangle.size(); // size (width, height) of the rectangle, 旋转之后最小边界框的宽高

    cv::Point2f points[4];
    rotatedRectangle.points(points); // The points array for storing rectangle vertices. The order is bottomLeft, topLeft, topRight, bottomRight.
                                     // 旋转之后的四个顶点，顺时针
    // cv::Mat depth_aligned_rect_pixels;
    cv::Mat depth_bounding_rect_rotated;

    cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);  //提取出最小边界框中的图像 
    // cv::getRectSubPix(depthimage, diagonal_rectangle_size, rotatedRectangle.center, depth_aligned_rect_pixels);

    cv::Point aligned_bounding_rect_center;

    aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;  // 最小边界框的中心
    aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;

    // 将提取出最小边界框进行旋转, 朝着反方向，因为要使图像变正， 所以这里的角度是 +90
    // 会有一部分显示不出来， 不过后面的cv::getRectSubPix() 会提取出中间的图像，刚好是旋转框旋转正之后的图像
    Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle+90, 1.0);
    cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);
    // cv::warpAffine(depth_aligned_rect_pixels, depth_bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);

    cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
    // cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);

    resized_rgb_crop = cv::Mat(cv::Size(resize_size, resize_size), CV_8UC1);
    cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);

    if(!rgb_cropped.empty()){
        vector<uchar> data_encode;
        imencode(".jpg", resized_rgb_crop, data_encode);

        std::string str_encode(data_encode.begin(), data_encode.end());
        char* char_r = new char[str_encode.size() + 10];
        memcpy(char_r, str_encode.data(), sizeof(char)*(str_encode.size()));
        return char_r;
    }
    else{
        return nullptr;
    }
} 


// 3.数据采集 --> 保存 RGB 和 depth
void save_rgb_depth(double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depthimage){
    int frame = global_frame;
    res_points.clear();
    std::vector<cv::Mat> channels;
    // cv::Mat depth_one_channel;
    // cv::split(depthimage, channels);
    // channels.at(0).convertTo(depth_one_channel, CV_32FC1);
    // cv::Mat depth_img = depth_one_channel;
    cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
    cv::Size size_of_initial_crop = cv::Size(rec_size * 2, rec_size * 2);  // 变成了两倍
    cv::Mat aligned_bounding_rect;
    cv::Mat Rotation_Matrix, bounding_rect_rotated;
    cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_32FC3); 

    // string depth_file_name = "/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/0623/0623_videos/frame/" + to_string(frame) + ".jpg";
    // cout<< "here .." <<endl;
    // cv::Mat depth_img = cv::imread(depth_file_name);

    // 旋转框
    cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle); //返回旋转后的矩形信息,  角度是逆负顺正
    cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect();  

    cv::Size rotatedRectangle_size = rotatedRectangle.size;  // width and height of the rectangle after rotation, 旋转后的宽、高和之前的一样
    cv::Size diagonal_rectangle_size = diagonal_rectangle.size();   

    cv::Point2f points[4];
    rotatedRectangle.points(points);   

    cv::cvtColor(source_image, source_image, CV_BGRA2BGR);   

    cv::Mat depth_aligned_rect_pixels;
    cv::Mat depth_bounding_rect_rotated;

    cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);  //提取出最小边界框中的图像 
    cv::getRectSubPix(depthimage, diagonal_rectangle_size, rotatedRectangle.center, depth_aligned_rect_pixels);  //here  !!!

    cv::Point aligned_bounding_rect_center;
    aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;  // 最小边界框的中心
    aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;

    if(res_points.empty()){
        res_points.emplace_back(points[0]);
        res_points.emplace_back(points[1]);
        res_points.emplace_back(points[2]);
        res_points.emplace_back(points[3]);
        // res_points.emplace_back(aligned_bounding_rect_center);
    }

    Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle+90, 1.0);
    cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);
    cv::warpAffine(depth_aligned_rect_pixels, depth_bounding_rect_rotated, Rotation_Matrix, depth_aligned_rect_pixels.size(), cv::INTER_CUBIC);

    cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
    cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);

    resized_rgb_crop = cv::Mat(cv::Size(resize_size, resize_size), CV_32FC3);
    // resized_depth_crop = cv::Mat(cv::Size(244,244), CV_8UC3);
    cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);
    cv::resize(depth_cropped, resized_depth_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);
    // cv::imshow("rgb", source_image);
    // cv::imshow("depth", depthimage);
    // cv::waitKey(33);
    string rgb_name = "./depth_and_rgb/cropped_img_" + to_string(frame) + ".jpg"; 
    string depth_name = "./depth_and_rgb/cropped_depth_" + to_string(frame) + ".png"; 
    cv::imwrite(rgb_name, resized_rgb_crop);
    cv::imwrite(depth_name, resized_depth_crop);
    frame++;
}


// 返回结构体（同时返回egb和depth，但是一直读取不到数据）
// croppedResult* crop_rgb_depth(double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depth_image){
//     res_points.clear();
//     cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
//     cv::Size size_of_initial_crop = cv::Size(rec_size * 2, rec_size * 2);  // 变成了两倍
//     cv::Mat aligned_bounding_rect;
//     cv::Mat Rotation_Matrix, bounding_rect_rotated;
//     cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_32FC3);
//     // 旋转框
//     cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle); //返回旋转后的矩形信息,  角度是逆负顺正
//     cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect(); // returns the minimal up-right integer rectangle containing the rotated rectangle,  这个返回的是旋转框的最小边界框
//     cv::Size rotatedRectangle_size = rotatedRectangle.size;  // width and height of the rectangle after rotation, 旋转后的宽、高和之前的一样
//     cv::Size diagonal_rectangle_size = diagonal_rectangle.size(); // size (width, height) of the rectangle, 旋转之后最小边界框的宽高
//     cv::Point2f points[4];
//     rotatedRectangle.points(points); // The points array for storing rectangle vertices. The order is bottomLeft, topLeft, topRight, bottomRight.
//                                      // 旋转之后的四个顶点，顺时针
//     cv::cvtColor(source_image, source_image, CV_BGRA2BGR);
//     // cv::Mat depth_aligned_rect_pixels;
//     cv::Mat depth_aligned_rect_pixels;
//     cv::Mat depth_bounding_rect_rotated;
//     cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);  //提取出最小边界框中的图像 
//     cv::getRectSubPix(depth_image, diagonal_rectangle_size, rotatedRectangle.center, depth_aligned_rect_pixels);
//     cv::Point aligned_bounding_rect_center;
//     aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;  // 最小边界框的中心
//     aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;
//     Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle+90, 1.0);
//     cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);
//     cv::warpAffine(depth_aligned_rect_pixels, depth_bounding_rect_rotated, Rotation_Matrix, depth_aligned_rect_pixels.size(), cv::INTER_CUBIC);
//     cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
//     cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);    
//     resized_rgb_crop = cv::Mat(cv::Size(resize_size,resize_size), CV_32FC3);
//     cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);
//     cv::resize(depth_cropped, resized_depth_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA); 
//     if(!rgb_cropped.empty() && !depth_cropped.empty()){
//         vector<uchar> data_encode, data_encode2;
//         imencode(".jpg", resized_rgb_crop, data_encode);
//         imencode(".jpg", resized_depth_crop, data_encode2);
//         std::string str_encode(data_encode.begin(), data_encode.end());
//         std::string str_encode2(data_encode2.begin(), data_encode2.end());
//         char* char_r = new char[str_encode.size() + 10];
//         char* char_r2 = new char[str_encode2.size() + 10];
//         memcpy(char_r, str_encode.data(), sizeof(char)*(str_encode.size()));
//         memcpy(char_r2, str_encode2.data(), sizeof(char)*(str_encode2.size()));
//         croppedResult* CRes = new croppedResult;
//         CRes->cropped_hand_depth = char_r2;
//         CRes->cropped_hand_rgb = char_r;
//         // cout<<"sucess .."<<endl;
//         return CRes;
//     }
//     else{
//         return nullptr;
//     }
// }

// 4.同时裁剪出RGB和depth，再分别赋值给两个全局变量，获取得到这两个全局变量就好了
void crop_rgb_depth(double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depth_image){
    res_points.clear();
    cv::Mat resized_rgb_crop, resized_depth_crop, depth_cropped;
    cv::Size size_of_initial_crop = cv::Size(rec_size * 2, rec_size * 2);  // 变成了两倍
    cv::Mat aligned_bounding_rect;
    cv::Mat Rotation_Matrix, bounding_rect_rotated;
    cv::Mat rgb_cropped = cv::Mat(size_of_initial_crop, CV_32FC3);

    // 旋转框
    cv::RotatedRect rotatedRectangle = cv::RotatedRect(joint_pt, size_of_initial_crop, angle); //返回旋转后的矩形信息,  角度是逆负顺正
    cv::Rect diagonal_rectangle = rotatedRectangle.boundingRect(); // returns the minimal up-right integer rectangle containing the rotated rectangle,  这个返回的是旋转框的最小边界框
    
    cv::Size rotatedRectangle_size = rotatedRectangle.size;  // width and height of the rectangle after rotation, 旋转后的宽、高和之前的一样
    cv::Size diagonal_rectangle_size = diagonal_rectangle.size(); // size (width, height) of the rectangle, 旋转之后最小边界框的宽高

    cv::Point2f points[4];
    rotatedRectangle.points(points); // The points array for storing rectangle vertices. The order is bottomLeft, topLeft, topRight, bottomRight.
                                     // 旋转之后的四个顶点，顺时针
    cv::cvtColor(source_image, source_image, CV_BGRA2BGR);

    // cv::Mat depth_aligned_rect_pixels;
    cv::Mat depth_aligned_rect_pixels;
    cv::Mat depth_bounding_rect_rotated;

    cv::getRectSubPix(source_image, diagonal_rectangle_size, rotatedRectangle.center, aligned_bounding_rect);  //提取出最小边界框中的图像 
    cv::getRectSubPix(depth_image, diagonal_rectangle_size, rotatedRectangle.center, depth_aligned_rect_pixels);

    cv::Point aligned_bounding_rect_center;
    aligned_bounding_rect_center.x = aligned_bounding_rect.cols / 2;  // 最小边界框的中心
    aligned_bounding_rect_center.y = aligned_bounding_rect.rows / 2;

    // 保存旋转后的点,用另外一个函数返回
    if(res_points.empty()){
        res_points.emplace_back(points[0]);
        res_points.emplace_back(points[1]);
        res_points.emplace_back(points[2]);
        res_points.emplace_back(points[3]);
        // res_points.emplace_back(aligned_bounding_rect_center);
    }

    Rotation_Matrix = getRotationMatrix2D(aligned_bounding_rect_center, angle+90, 1.0);
    cv::warpAffine(aligned_bounding_rect, bounding_rect_rotated, Rotation_Matrix, aligned_bounding_rect.size(), cv::INTER_CUBIC);
    cv::warpAffine(depth_aligned_rect_pixels, depth_bounding_rect_rotated, Rotation_Matrix, depth_aligned_rect_pixels.size(), cv::INTER_CUBIC);

    cv::getRectSubPix(bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, rgb_cropped);
    cv::getRectSubPix(depth_bounding_rect_rotated, rotatedRectangle_size, aligned_bounding_rect_center, depth_cropped);    

    resized_rgb_crop = cv::Mat(cv::Size(resize_size, resize_size), CV_32FC3);
    cv::resize(rgb_cropped, resized_rgb_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA);
    cv::resize(depth_cropped, resized_depth_crop, resized_rgb_crop.size(), 0, 0, CV_INTER_AREA); 

    if(!rgb_cropped.empty() && !depth_cropped.empty()){
        vector<uchar> data_encode, data_encode2;
        imencode(".jpg", resized_rgb_crop, data_encode);
        imencode(".jpg", resized_depth_crop, data_encode2);

        std::string str_encode(data_encode.begin(), data_encode.end());
        std::string str_encode2(data_encode2.begin(), data_encode2.end());

        // char* char_r = new char[str_encode.size() + 10];
        // char* char_r2 = new char[str_encode2.size() + 10];
        // memcpy(char_r, str_encode.data(), sizeof(char)*(str_encode.size()));
        // memcpy(char_r2, str_encode2.data(), sizeof(char)*(str_encode2.size()));
        // cropped_rgb_global = char_r;
        // cropped_depth_global = char_r2;

        cropped_rgb_global = new char[str_encode.size() + 10];
        cropped_depth_global = new char[str_encode2.size() + 10];
        memcpy(cropped_rgb_global, str_encode.data(), sizeof(char)*(str_encode.size()));
        memcpy(cropped_depth_global, str_encode2.data(), sizeof(char)*(str_encode2.size()));
    }
}

// 包装成C函数
extern "C"{
    char* cdraw_rectangle(int height, int width, uchar* frame_data, int center_x, int center_y, int rec_size, double angle){
        int count = 0;
        cv::Point center_pt(center_x, center_y);

        Mat image(height, width, CV_8UC3, frame_data);  // 使用构造函数，速度快    
        auto tmp = draw_rectangle(angle, rec_size, center_pt, image, image);
        if(tmp != nullptr)
          return tmp;
    }

    char* cdraw_rectangle_depth(int height, int width, uchar* frame_data, int center_x, int center_y, int rec_size, double angle){
        int count = 0;
        cv::Point center_pt(center_x, center_y);

        Mat image(height+2, width, CV_8UC1, frame_data);  // 使用构造函数，速度快    
        auto tmp = draw_rectangle_depth(angle, rec_size, center_pt, image, image);
        if(tmp != nullptr)
          return tmp;
    }   

    void cget_rectangle_points(int pts[]){
        if(res_points.empty()){
            pts[0] = -35;
        }
        else{
            pts[0] = res_points[0].x;
            pts[1] = res_points[0].y;
            pts[2] = res_points[1].x;
            pts[3] = res_points[1].y;
            pts[4] = res_points[2].x;
            pts[5] = res_points[2].y;
            pts[6] = res_points[3].x;
            pts[7] = res_points[3].y;
            // pts[8] = res_points[4].x;
            // pts[9] = res_points[4].y;
        }
    }

    void csave_rgb_depth(int height, int width, uchar* rgb_frame_data, uchar* depth_frame_data, int center_x, int center_y, int rec_size, double angle){
        int count = 0;
        cv::Point center_pt(center_x, center_y);

        Mat image(height, width, CV_8UC3, rgb_frame_data);  // CV_8UC3 --> 8 bit, u unsingned int, C 所存储的通道数, 3代表彩色图 1 代表灰度图
        Mat depth(height+2, width, CV_8UC3, depth_frame_data);
            
        save_rgb_depth(angle, rec_size, center_pt, image, image, depth);     
    }

    // 将crop depth 和 crop rgb 作为全局变量返回
    void ccrop_rgb_depth(int height, int width, uchar* rgb_frame_data, uchar* depth_frame_data, int center_x, int center_y, int rec_size, double angle, int frame, int person_num, int dual_hand){
        cv::Point center_pt(center_x, center_y);
        if (frame == global_frame){
            crop_rgb_depth(angle, rec_size, center_pt, global_rgb, global_rgb, global_depth); // double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depth_image
        }
        else if (frame != global_frame){
            global_frame = frame;
            Mat image(height, width, CV_8UC3, rgb_frame_data);
            Mat depth(height+2, width, CV_8UC1, depth_frame_data);
            crop_rgb_depth(angle, rec_size, center_pt, image, image, depth); // double angle, int rec_size, cv::Point joint_pt, cv::Mat input_image, cv::Mat source_image, cv::Mat depth_image
            // 人数大于1的时候才会进行复制，进入第一个if,就少了复制图像的一步
            if (person_num > 1 || dual_hand){
                image.copyTo(global_rgb);
                depth.copyTo(global_depth);
            }
            // cout << "rgb: " << image.cols << " " << image.rows << " " << image.channels() << endl;
            // cout << "depth: " << depth.cols << " " << depth.rows << " " << depth.channels() << endl;
            // cv::imshow("cpp_rgb", image);
            // cv::imshow("cpp_depth", depth);
        }
        
        // cv::imshow("cpp_rgb", global_rgb);
        // cv::imshow("cpp_depth", global_depth);
        // cv::waitKey(1);

    }

    char* cget_cropped_rgb(){
        if(cropped_rgb_global)
        {
            return cropped_rgb_global;
        }
    }

    char* cget_cropped_depth(){
        if(cropped_depth_global)
        {
            return cropped_depth_global;
        }
    }


    void cdelete(){
        if(cropped_rgb_global){
            delete[] cropped_rgb_global;
            cropped_rgb_global = nullptr;
        }
        if(cropped_depth_global){
            delete[] cropped_depth_global;
            cropped_depth_global = nullptr;
        }
    }
}
