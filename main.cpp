#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

#include "C:\MinGW\include\boost\boost_1_79_0\boost\asio\detail\mutex.hpp"
using namespace std;
using namespace cv;
using namespace cv::dnn;


using namespace std;
using namespace cv;
using namespace cv::dnn;

//手部关键点数
const int nPoints = 21;
//手指索引
const int tipIds[] = { 4,8,12,16,20 };

//手部关键点检测
bool HandKeypoints_Detect(Mat src, vector<Point>&HandKeypoints)
{
    //模型尺寸大小
    int width = src.cols;
    int height = src.rows;
    float ratio = width / (float)height;
    int modelHeight = 368;  //由模型输入维度决定
    int modelWidth = int(ratio*modelHeight);

    //模型文件
    string model_file = "pose_deploy.prototxt";  //网络模型
    string model_weight = "pose_iter_102000.caffemodel";//网络训练权重

    //加载caffe模型
    Net net = readNetFromCaffe(model_file, model_weight);

    //将输入图像转成blob形式
    Mat blob = blobFromImage(src, 1.0 / 255, Size(modelWidth, modelHeight), Scalar(0, 0, 0));

    //将图像转换的blob数据输入到网络的第一层“image”层，见deploy.protxt文件
    net.setInput(blob, "image");

    //结果输出
    Mat output = net.forward();
    int H = output.size[2];
    int W = output.size[3];

    for (int i = 0; i < nPoints; i++)
    {
        //结果预测
        Mat probMap(H, W, CV_32F, output.ptr(0, i));

        resize(probMap, probMap, Size(width, height));

        Point keypoint; //最大可能性手部关键点位置
        double classProb;  //最大可能性概率值
        minMaxLoc(probMap, NULL, &classProb, NULL, &keypoint);

        HandKeypoints[i] = keypoint; //结果输出，即手部关键点所在坐标
    }

    return true;
}


//手势识别
bool Handpose_Recognition(vector<Point>&HandKeypoints, int& count)
{
    vector<int>fingers;
    //拇指
    if (HandKeypoints[tipIds[0]].x > HandKeypoints[tipIds[0] - 1].x)
    {
        //如果关键点'4'的x坐标大于关键点'3'的x坐标，则说明大拇指是张开的。计数1
        fingers.push_back(1);
    }
    else
    {
        fingers.push_back(0);
    }
    //其余的4个手指
    for (int i = 1; i < 5; i++)
    {
        if (HandKeypoints[tipIds[i]].y < HandKeypoints[tipIds[i] - 2].y)
        {
            //例：如果关键点'8'的y坐标小于关键点'6'的y坐标，则说明食指是张开的。计数1
            fingers.push_back(1);
        }
        else
        {
            fingers.push_back(0);
        }
    }

    //结果统计
    for (int i = 0; i < fingers.size(); i++)
    {
        if (fingers[i] == 1)
        {
            count++;
        }
    }

    return true;
}


//识别效果显示
bool ShowResult(Mat& src, vector<Point>&HandKeypoints, int& count)
{
    //画出关键点所在位置
    for (int i = 0; i < nPoints; i++)
    {
        circle(src, HandKeypoints[i], 3, Scalar(0, 0, 255), -1);
        putText(src, to_string(i), HandKeypoints[i], FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 255, 0), 2);
    }

    //为了显示骚操作，读取模板图片，作为识别结果
    vector<string>imageList;
    string filename = "images/";
    glob(filename, imageList);

    vector<Mat>Temp;
    for (int i = 0; i < imageList.size(); i++)
    {
        Mat temp = imread(imageList[i]);

        resize(temp, temp, Size(100, 100), 1, 1, INTER_AREA);

        Temp.push_back(temp);
    }

    //将识别结果显示在原图中
    Temp[count].copyTo(src(Rect(0, src.rows- Temp[count].rows, Temp[count].cols, Temp[count].rows)));
    putText(src, to_string(count), Point(20, 60), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 128), 3);

    return true;
}

//将所有图片整合成一张图片
bool Stitching_Image(vector<Mat>images)
{
    Mat canvas = Mat::zeros(Size(1200, 1000), CV_8UC3);
    int width = 400;
    int height = 500;

    for (int i = 0; i < images.size(); i++)
    {
        resize(images[i], images[i], Size(width, height), 1, 1, INTER_LINEAR);
    }

    int col = canvas.cols / width;
    int row = canvas.rows / height;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int index = i * col + j;
            images[index].copyTo(canvas(Rect(j*width, i*height, width, height)));
        }
    }

    namedWindow("result", WINDOW_NORMAL);
    imshow("result", canvas);
    waitKey(0);
    return true;
}



int main()
{
    vector<string>imageList;
    string filename = "test/";
    glob(filename, imageList);

    vector<Mat>images;
    for (int i = 0; i < imageList.size(); i++)
    {
        Mat src = imread(imageList[i]);

        vector<Point>HandKeypoints(nPoints);
        HandKeypoints_Detect(src, HandKeypoints);

        int count = 0;
        Handpose_Recognition(HandKeypoints, count);

        ShowResult(src, HandKeypoints, count);
        images.push_back(src);

        imshow("Demo", src);
        waitKey(0);
    }

    Stitching_Image(images);

    system("pause");
    return 0;
}


