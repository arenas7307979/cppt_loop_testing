#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include "basic_datatype/frame.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include "basic_datatype/util_datatype.h"
#include "../reloc/3rdParty/DBoW2/DBoW2/DBoW2.h"
#include "../reloc/3rdParty/DBoW2/DBoW2/BowVector.h"
#include "basic_datatype/keyframe.h"
#include "camera_model/simple_stereo_camera.h"
#include <opencv2/core/eigen.hpp>
#include "ceres/ceres.h"


class SimplePoseGraph{
public:
    enum PoseGraphyState {
        INIT,
        NON_LINEAR
    };
    SimplePoseGraph(const SimpleStereoCamPtr& camera_);
    ~SimplePoseGraph();
    void Process();

    //add keyframe from BackEnd
    void AddKeyFrameAfterBA(const FramePtr& frame);

    void SetPoseGraphCallback(const std::function<void(const FramePtr keyframe)>& PG_callback);

    std::function<void(const FramePtr keyframe)> mPoseGraphCallback;

    //Pose graph optimizer
    void optimize6DoF();

    PoseGraphyState mState;

private:
    SimpleStereoCamPtr mpCamera;

    bool LoadORBVocabulary(std::string& voc_path);
    int  LoopDetection(const KeyframePtr& keyframe, const int& db_id);

    //find connection between cur_keyframe and old_keyframe(from dbow query)
    bool FindKFConnect(KeyframePtr& cur_kf, KeyframePtr& old_kf);

    //find the best maching and save correspondence from each old_kfs desc
    bool SearchInAera(const cv::Mat &window_descriptor_cur,
                      const std::vector<cv::Mat> &descriptors_old,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      cv::Point2f &best_match_old, int &idx);

    void MatchKeyFrame(KeyframePtr& cur_kf, KeyframePtr& old_kf,
                       std::vector<cv::Point2f> &matched_2d_cur,
                       std::vector<cv::Point2f> &matched_2d_old,
                       std::vector<cv::Point3f> &matched_x3Dw);

    //for remove minority number of keypoint angle
    void ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

    //estimate ORB desc hamming distance
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    //index in the loop
    int db_index = 0;
    int earliest_loop_index = -1;

    //Drift is optimize6DoF reult,
    //represent drift between original path and after loop-optimize path
    Sophus::SE3d T_drift;

    //Pose graph optimizer may slower than loop-detection
    //that for save keyframe index.
    std::vector<int> optimize_buf;
    std::mutex m_optimize_buf;
    std::condition_variable mOptimizeBuffer;

    //check frame
    Eigen::Vector3d last_t;

    //mutex to protect pose graph process
    std::mutex pg_process;
    std::mutex mKFBufferMutex;
    std::vector<FramePtr> mKFBuffer;
    std::condition_variable mKFBufferPG;

    //for vocabulary
    OrbVocabulary* ORBVocabulary;
    OrbDatabase ORBdatabase;

    //for keyframe IC angle
    std::vector<int> umax;

    //save keyframes info in pose graph.
    map<int, KeyframePtr> keyframes;
    std::mutex keyframesMutex;

#if DEBUG_POSEGRAPH
    //for debug
    map<int, cv::Mat> kf_images;
#endif
};

struct SixDOFError
{
    SixDOFError(Sophus::SE3d relativePose)
        :relativePose(relativePose){}

    template <typename T> // Tw1                         Tw2
    bool operator()(const T*  TiMinusj_, const T* Ti_ , T* residuals) const
    {
        Eigen::Map<const Sophus::SE3<T>> TiMinusj(TiMinusj_);
        Eigen::Map<const Sophus::SE3<T>> Ti(Ti_);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
        Sophus::SE3<T> new_relative_pose = TiMinusj.inverse() * Ti; // T12
        Sophus::SE3<T> residual_T =  relativePose.inverse() * new_relative_pose ;
        Eigen::Quaternion<T> q = residual_T.unit_quaternion();
        residual.head(3) = residual_T.translation();
        residual.tail(3) << (q.x(), q.y(), q.z());

        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d relativePose)
    {
        return (new ceres::AutoDiffCostFunction<SixDOFError, 6, 7, 7>(
                    new SixDOFError(relativePose)));
    }
    Sophus::SE3d relativePose; // T12
};

SMART_PTR(SimplePoseGraph)
