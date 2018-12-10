#pragma once
#include <memory>
#include <vector>
#include <sophus/se3.hpp>
#include "util_datatype.h"
#include "mappoint.h"
#include "frame.h"
#define DEBUG_POSEGRAPH 1

class MapPoint;
SMART_PTR(MapPoint)

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

class Keyframe : public std::enable_shared_from_this<Keyframe> {
public:

    Keyframe(const FramePtr frame,  std::vector<int>& umax);
    ~Keyframe();

    void computeORBDescriptors(const FramePtr frame);
    uint64_t mKeyFrameID;
    Sophus::SE3d mTwc;

    // image points
    std::vector<uint32_t> mvPtCount;
    std::vector<cv::Point2f> mv_uv;
    std::vector<float> mv_ur; // value -1 is mono point
    std::vector<MapPointPtr> mvMapPoint; // FIXME
    uint32_t mNumStereo;

#if DEBUG_POSEGRAPH
    cv::Mat mImgL;
#endif

    //Image desc/points
    std::vector<cv::Mat> descriptors;
    std::vector<cv::KeyPoint> k_pts;
    double mTimeStamp;

private:
    void changeORBdescStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);
};

SMART_PTR(Keyframe)
