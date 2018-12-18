#pragma once
#include <memory>
#include <vector>
#include <sophus/se3.hpp>
#include "util_datatype.h"
#include "mappoint.h"
#include "frame.h"
#include <opencv2/core/eigen.hpp>

#define DEBUG_POSEGRAPH 1
class MapPoint;
SMART_PTR(MapPoint)

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

class Keyframe : public std::enable_shared_from_this<Keyframe> {
public:

    Keyframe(const FramePtr frame,  std::vector<int>& umax, int frame_index);
    ~Keyframe();

    void computeORBDescriptors(const FramePtr frame);

    //save MatchKeyframe info (relative pose) in pose graph
    void setRelativeInfo(Sophus::SE3d &relative_T_, bool &has_loop_, double &relative_yaw_ ,int Match_loop_index_);

    void getRelativeInfo(Sophus::SE3d &relative_T_, double &relative_yaw_ ,int Match_loop_index);

    void updateVioPose(Sophus::SE3d &mTwc_loop);

    int mKeyFrameID;
    Sophus::SE3d mTwc;

    //save index in pose graph, local index in 4DOF/6DOF optimizer
    int indexInLoop = 0;
    int local_index= 0;
    // image points Iinfo
    std::vector<uint32_t> mvPtCount;
    std::vector<cv::Point2f> mv_uv;
    std::vector<float> mv_ur; // value -1 is mono point
    std::vector<Eigen::Vector3d> x3Dws;
    uint32_t mNumStereo;

#if DEBUG_POSEGRAPH
    cv::Mat mImgL;
#endif

    //Image desc/points
    std::vector<cv::Mat> descriptors;
    std::vector<cv::KeyPoint> k_pts;
    double mTimeStamp;

    // ceres solver to optimize pose
    double c_rotation[4];
    double c_translation[3];
    double vertex_data[7];

    // have been finding loop
    bool has_loop;
private:
    void changeORBdescStructure(const cv::Mat &plain, std::vector<cv::Mat> &out);

    //save MatchKeyframe info (relative pose) in pose graph
    int Match_loop_index=-1;
    //Tc2c1, c2::Loop pose rom old keyframe info, c1:: cur pose from VO/VIO
    Sophus::SE3d relative_T;
    double relative_yaw;
};

SMART_PTR(Keyframe)
