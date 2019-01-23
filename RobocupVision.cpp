#include "RobocupVision.h"

RobocupVision::RobocupVision() {
    start_file_num_ = 0;
    max_file_num_   = 500;
    LoadEverything();
}

void RobocupVision::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    pretreated_image_    = Pretreat(input_image);

    // pix thre
    glass_binary_image   = ProcessGlassColor(pretreated_image_);
    ball_binary_image    = ProcessBallColor(pretreated_image_);

    // fit the sideline discrete points by least quares method
    cv::Mat mat_a(slide_window_num_, 2, CV_64FC1);
    cv::Mat mat_x(2, 1, CV_64FC1);
    cv::Mat mat_b(slide_window_num_, 1, CV_64FC1);
    sideline_border_discrete_points_ = GetSideLineBySldWin(glass_binary_image);
    for (int i = 0; i < slide_window_num_; i++) {
        mat_a.at<double>(i, 0) = sideline_border_discrete_points_[i].x;
        mat_a.at<double>(i, 1) = 1.;

        mat_b.at<double>(i, 0) = sideline_border_discrete_points_[i].y;
    }
    cv::Mat mat_a_t = mat_a.t();
    mat_x = (mat_a_t*mat_a).inv()*mat_a_t*mat_b;

    // Judge the sideline result
    if (fabs(mat_x.at<double>(0, 0)) > -1) {    // it should be stable (￣▽￣)""
        final_result_.sideline_valid_   = true;
        final_result_.sideline_slope_   = mat_x.at<double>(0, 0);
        final_result_.sideline_center_  = cv::Point2i(src_image_.cols/2, 
                                                    mat_x.at<double>(0, 0)*src_image_.cols/2.+mat_x.at<double>(1, 0));
        final_result_.sideline_begin_   = cv::Point2i(0, 
                                                    mat_x.at<double>(1, 0));
        final_result_.sideline_end_     = cv::Point2i(src_image_.cols, 
                                                    mat_x.at<double>(0, 0)*src_image_.cols+mat_x.at<double>(1, 0));
    }

    // Get Ball Relate within the line area
    ball_possible_rects_ = GetPossibleBallRect(ball_binary_image);
    // feed ball possible rect to ball classifier 
    std::vector<cv::Rect> ball_pos_lable_rects;
    for (std::vector<cv::Rect>::iterator iter = ball_possible_rects_.begin();
         iter != ball_possible_rects_.end(); iter++) {
        cv::Mat roi_hog_vec = GetHogVec(*iter);
        int lable = (int)ball_classifier_.predict(roi_hog_vec);
        if (lable == Ball_POS) {
            ball_pos_lable_rects.push_back(*iter);
        }
    }

    // Judge the ball result
    if (ball_pos_lable_rects.size() == 0) {
        final_result_.ball_valid_ = false;
    }
    else if (ball_pos_lable_rects.size() == 1) {
        final_result_.ball_valid_ = true;
        ball_result_rect_ = ball_pos_lable_rects[0];
        final_result_.ball_center_ = cv::Point2d(ball_result_rect_.x + cvRound(ball_result_rect_.width/2),
                                                 ball_result_rect_.y + cvRound(ball_result_rect_.height/2));
    }
    else {
        final_result_.ball_valid_ = false;
    }

    (*dynamic_cast<RobocupResult*>(output_result)) = final_result_;
#ifndef ADJUST_PARAMETER
    WriteImg(src_image_, "src_img", start_file_num_);
    if (final_result_.sideline_valid_) {
        cv::line(src_image_, final_result_.sideline_begin_, final_result_.sideline_end_, cv::Scalar(0, 0, 255), 3);
    }
    if (final_result_.ball_valid_) {
        cv::rectangle(src_image_, ball_result_rect_, cv::Scalar(0, 255, 0));
    }
    WriteImg(src_image_, "center_img", start_file_num_++);
#endif 
}

