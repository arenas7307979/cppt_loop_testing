#include "simple_frontend.h"
#include "basic_datatype/tic_toc.h"

template<class T>
void ReduceVector(std::vector<T> &v, const std::vector<uchar>& status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

bool InBorder(const cv::Point2f& pt, int width, int height) {
    const int border_size = 1;
    int img_x = std::round(pt.x);
    int img_y = std::round(pt.y);
    if(img_x - border_size < 0 || img_y - border_size <= 0 || img_x + border_size >= width ||
            img_y + border_size >= height)
        return false;
    return true;
}

SimpleFrontEnd::SimpleFrontEnd(const SimpleStereoCamPtr& camera)
    : mpCamera(camera), mFeatureID(0)
{

}

SimpleFrontEnd::~SimpleFrontEnd() {

}

// with frame without any feature extract by FAST corner detector
// if exist some features extract FAST corner in empty grid.
void SimpleFrontEnd::ExtractFeatures(FramePtr frame) {
    static int empty_value = -1, exist_value = -2;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height)/32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width)/32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = frame->mv_uv.size(); i < n; ++i) {
        auto& pt = frame->mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grids[grid_i][grid_j] != exist_value)
            grids[grid_i][grid_j] = exist_value;
    }

    std::vector<cv::KeyPoint> kps;
    cv::FAST(frame->mImgL, kps, 13);

    for(int i = 0, n = kps.size(); i < n; ++i) {
        auto& pt = kps[i].pt;
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grid_i >= grid_rows)
            grid_i = grid_rows - 1;
        if(grid_j >= grid_cols)
            grid_j = grid_cols - 1;
        int value = grids[grid_i][grid_j];
        if(value != exist_value) {
            if(value == empty_value) {
                grids[grid_i][grid_j] = i;
            }
            else {
                if(kps[i].response > kps[value].response)
                    grids[grid_i][grid_j] = i;
            }
        }
    }

    for(auto& i : grids)
        for(auto& j : i)
            if(j >= 0) {
                frame->mv_uv.emplace_back(kps[j].pt);
                frame->mvPtID.emplace_back(mFeatureID++);
                frame->mvPtCount.emplace_back(0);
            }

    cv::Mat result;
    cv::cvtColor(frame->mImgL, result, CV_GRAY2BGR);
    for(auto& pt : frame->mv_uv) {
        cv::circle(result, pt, 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("a", result);
    cv::waitKey(1);
}

// track features by optical flow and check epipolar constrain
void SimpleFrontEnd::TrackFeaturesByOpticalFlow(FramePtr ref_frame, FramePtr cur_frame) {
    const int width = mpCamera->width, height = mpCamera->height;

    // copy ref frame data for cur frame
    std::vector<uint64_t> ref_frame_pt_id = ref_frame->mvPtID;
    std::vector<uint32_t> ref_frame_pt_count = ref_frame->mvPtCount;
    std::vector<cv::Point2f> ref_frame_pts = ref_frame->mv_uv;

    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(ref_frame->mImgL, cur_frame->mImgL, ref_frame_pts,
                             cur_frame->mv_uv, status, err, cv::Size(21, 21), 3);

    for(int i = 0, n = cur_frame->mv_uv.size(); i < n; ++i)
        if(status[i] && !InBorder(cur_frame->mv_uv[i], width, height))
            status[i] = 0;

    ReduceVector(ref_frame_pt_id, status);
    ReduceVector(ref_frame_pt_count, status);
    ReduceVector(ref_frame_pts, status);
    ReduceVector(cur_frame->mv_uv, status);

    RemoveOutlierFromF(ref_frame_pt_id, ref_frame_pt_count, ref_frame_pts, cur_frame->mv_uv);

    // uniform the feature distribution
    UniformFeatureDistribution(ref_frame_pt_id, ref_frame_pt_count, cur_frame->mv_uv);

    // copy to cur_frame
    cur_frame->mvPtID = ref_frame_pt_id;
    cur_frame->mvPtCount = ref_frame_pt_count;
    for(auto& it : cur_frame->mvPtCount)
        ++it;
}

// simple sparse stereo matching algorithm by optical flow
void SimpleFrontEnd::SparseStereoMatching(FramePtr frame) {

}

void SimpleFrontEnd::RemoveOutlierFromF(std::vector<uint64_t>& ref_pt_id,
                                        std::vector<uint32_t>& ref_pt_count,
                                        std::vector<cv::Point2f>& ref_pts,
                                        std::vector<cv::Point2f>& cur_pts) {
    if(cur_pts.size() > 8) {
        std::vector<uchar> status;
        cv::findFundamentalMat(ref_pts, cur_pts, cv::FM_RANSAC, 1, 0.99, status);
        ReduceVector(ref_pt_id, status);
        ReduceVector(ref_pt_count, status);
        ReduceVector(ref_pts, status);
        ReduceVector(cur_pts, status);
    }
}

void SimpleFrontEnd::UniformFeatureDistribution(std::vector<uint64_t>& ref_pt_id,
                                                std::vector<uint32_t>& ref_pt_count,
                                                std::vector<cv::Point2f>& cur_pts) {
    static int empty_value = -1;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height)/32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width)/32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = cur_pts.size(); i < n; ++i) {
        auto& pt = cur_pts[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        int idx = grids[grid_i][grid_j];
        if(idx != empty_value) {
            if(ref_pt_count[i] > ref_pt_count[idx])
                grids[grid_i][grid_j] = i;
        }
        else
            grids[grid_i][grid_j] = i;
    }

    std::vector<uint64_t> temp_pt_id;
    std::vector<uint32_t> temp_pt_count;
    std::vector<cv::Point2f> temp_pts;
    for(auto& i : grids) {
        for(auto& j : i) {
            if(j != empty_value) {
                temp_pt_id.emplace_back(ref_pt_id[j]);
                temp_pt_count.emplace_back(ref_pt_count[j]);
                temp_pts.emplace_back(cur_pts[j]);
            }
        }
    }

    ref_pt_id = std::move(temp_pt_id);
    ref_pt_count = std::move(temp_pt_count);
    cur_pts = std::move(temp_pts);
}
