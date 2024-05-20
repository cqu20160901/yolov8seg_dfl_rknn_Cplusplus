#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include <opencv2/highgui.hpp>

typedef signed char int8_t;
typedef unsigned int uint32_t;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int classId;
    float mask[32];
} DetectRect;

// yolov8
class GetResultRectYolov8seg
{
public:
    GetResultRectYolov8seg();

    ~GetResultRectYolov8seg();

    int GenerateMeshGrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects, cv::Mat &SegMask);

    float sigmoid(float x);

private:
    std::vector<float> MeshGrid;

    const int ClassNum = 80;
    int HeadNum = 3;

    int InputWidth = 640;
    int InputHeight = 640;
    int Strides[3] = {8, 16, 32};
    int MapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    int MaskNum = 32;
    int SegWidth = 160;
    int SegHeight = 160;

    float NmsThresh = 0.45;
    float ObjectThresh = 0.5;

    std::vector<float> RegDfl;
    float RegDeq[16] = {0};

    std::vector<cv::Vec3b> ColorLists = {cv::Vec3b(000, 000, 255),
                                         cv::Vec3b(255, 128, 000),
                                         cv::Vec3b(255, 255, 000),
                                         cv::Vec3b(000, 255, 000),
                                         cv::Vec3b(000, 255, 255),
                                         cv::Vec3b(255, 000, 000),
                                         cv::Vec3b(128, 000, 255),
                                         cv::Vec3b(255, 000, 255),
                                         cv::Vec3b(128, 000, 000),
                                         cv::Vec3b(000, 128, 000)};
};

#endif
