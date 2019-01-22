#ifndef ROBOCUPVISION_H
#define ROBOCUPVISION_H

// Lable for SVM Glassifier
#define Ball_POS 1
#define Ball_NEG 0

#define ADJUST_PARAMETER

#include <opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER // need asj

// showing image in debugging 
#define SHOW_IMAGE(imgName) \
    namedWindow("imgName", WINDOW_AUTOSIZE); \
    moveWindow("imgName", 300, 300); \
    imshow("imgName", imgName); \
    waitKey(5000); \
    destroyWindow("imgName"); \


class ImgProcResult{

public:
    ImgProcResult(){};
    ~ImgProcResult(){};
    virtual void operator=(ImgProcResult &res) = 0;
private:
protected:
    ImgProcResult* res;
};

class ImgProc{

public:
    ImgProc(){};
    ~ImgProc(){};
    virtual void imageProcess(cv::Mat img, ImgProcResult *Result) =0;
private:
protected:
    ImgProcResult *res;

};
#else

#include "imgproc.h"
#define SHOW_IMAGE(imgName) ;

#endif 

class RobocupResult : public ImgProcResult {
public: // data menber
    // sideline detection relate
    bool sideline_valid_;
    double sideline_slope_;
    cv::Point2d sideline_center_;

    // ball detection relate
    bool ball_valid_;
    cv::Point2d ball_center_;

    // goal detection relate
    bool goal_valid_;
    cv::Point2d goal_center_;

    // robo detection relate


    // location relate
public:
    RobocupResult() {
        sideline_valid_ = false;
        ball_valid_     = false;
        goal_valid_     = false; 
    }

    virtual void operator=(ImgProcResult& res) {
        RobocupResult* tmp  = dynamic_cast<RobocupResult*>(&res);
        
        sideline_valid_     = tmp->sideline_valid_;
        sideline_slope_     = tmp->sideline_slope_;
        sideline_center_    = tmp->sideline_center_; 

        ball_valid_         = tmp->ball_valid_;
        ball_center_        = tmp->ball_center_;

        goal_valid_         = tmp->goal_valid_;
        goal_center_        = tmp->goal_center_;
    }

    void operator=(RobocupResult& res) {
        sideline_valid_     = res.sideline_valid_;
        sideline_slope_     = res.sideline_slope_;
        sideline_center_    = res.sideline_center_;

        ball_valid_         = res.ball_valid_;
        ball_center_        = res.ball_center_;

        goal_valid_         = res.goal_valid_;
        goal_center_        = res.goal_center_;
    }
};

struct AllParameters {

};

class RobocupVision : public ImgProc {
public:
    RobocupVision();

public:
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // external interface
    
    cv::Mat Pretreat(cv::Mat raw_image);                                    // all pretreatment, image enhancement for the src_image and etc

    cv::Mat ProcessGlassColor(cv::Mat pretreated_image);                    // get the Glass binary image

    cv::Mat ProcessBallColor(cv::Mat pretreated_image);                     // get the ball binary image

    void GetSideLineBySldWin(cv::Mat binary_image);                         // get the rough sideline by using slide windows and least squares fit

    std::vector<cv::Rect> GetPossibleBallRect(cv::Mat binary_image);         // get the possible ball's rects in the ball binary image with the help of sideline

    cv::Mat GetHogVec(cv::Rect roi);                                        // get the hog feature vector of roi in src_img 

public:
    void LoadEverything();                                                  // load parameters from the file AS WELL AS the SVM MODEL !!!!

    void StoreParameters();                                                 // Store parameters to file

    void set_all_parameters(AllParameters ap);                              // when setting parameters in main.cpp

    void WriteImg(cv::Mat src, string folder_name, int num);                // while running on darwin, save images

public: // data menbers
    // father of everything
    cv::Mat src_image_;

    // for Pretreat
    cv::Mat src_hls_channels_[3];
    cv::Mat src_hsv_channels_[3];
    cv::Mat pretreated_image_;      // the after-enhancement src image 
    
    // for ProcessXColor
    cv::Mat glass_binary_image;
    int glass_h_min_thre_;
    int glass_h_max_thre_;
    int glass_h_direction_forward_;
    int glass_erode_times_;
    int glass_dilate_times_;
    cv::Mat ball_binary_image;
    int ball_l_min_thre_;
    int ball_l_max_thre_;

    // for GetSideLineBySldWin
    int slide_window_num_;
    int slide_window_cols_;
    int slide_window_rows_;
    std::vector<cv::Rect> slide_wins_;
    int slide_stride_;
    double slide_win_thre_rate_;

    // for GetPossibleBallRect
    std::vector<cv::Rect> ball_possible_rects_;
    cv::Rect ball_result_rect_;
    int ball_rect_area_thre_;

    // for GetHogVec
    // null
    // all can be motified in the function body (*^_^*)
    // because it is seldom motified, which need re-train the whole svm classifier

    // for WriteImg
    int start_file_num_;
    int max_file_num_;

    // for SVM classifier
    CvSVM ball_classifier_;
    string svm_model_name_;     

    // result & etc
    RobocupResult final_result_;
};  

#endif