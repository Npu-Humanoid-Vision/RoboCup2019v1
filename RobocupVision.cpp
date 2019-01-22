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

    // Get line
    GetSideLineBySldWin(glass_binary_image);

    // Judge the sideline result
    

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
        final_result_.ball_center_ = cv::Point2d(ball_result_rect_.x + cvRound(ball_result_rect_.width/2.),
                                                 ball_result_rect_.y + cvRound(ball_result_rect_.height/2.));
    }
    else {
        final_result_.ball_valid_ = false;
    }

    (*dynamic_cast<RobocupResult*>(output_result)) = final_result_;
#ifndef ADJUST_PARAMETER
    WriteImg(src_image_, "src_img", start_file_num_);
    if (final_result_.ball_valid_) {
        cv::rectangle(src_image_, ball_result_rect_, cv::Scalar(0, 255, 0));
    }
    WriteImg(src_image_, "center_img", start_file_num_++);
#endif 
}