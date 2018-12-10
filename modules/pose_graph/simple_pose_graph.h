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
#include "basic_datatype/util_datatype.h"

class SimplePoseGraph{
public:
    enum PoseGraphyState {
        INIT,
        NON_LINEAR
    };
    SimplePoseGraph(const SimpleStereoCamPtr& camera_);
    ~SimplePoseGraph();
    void Process();
    void AddKeyFrameAfterBA(const FramePtr& frame);
    void SetPoseGraphCallback(const std::function<void(const FramePtr keyframe)>& PG_callback);
    PoseGraphyState mState;
    std::function<void(const FramePtr keyframe)> mPoseGraphCallback;

private:
    SimpleStereoCamPtr mpCamera;

    bool LoadORBVocabulary(std::string& voc_path);
    int  LoopDetection(const KeyframePtr& keyframe, const int& db_id);

    //find connection between cur_keyframe and old_keyframe(from dbow query)
    bool FindKFConnect(KeyframePtr& cur_kf, KeyframePtr& old_kf);

    //Descriptor distance for compare similarity(hamming)
    bool searchInAera(const cv::Mat &window_descriptor_cur,
                      const std::vector<cv::Mat> &descriptors_old,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      cv::Point2f &best_match);


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
    int db_index = 0;

    //for keyframe IC angle
    std::vector<int> umax;
    map<int, KeyframePtr> keyframes;

#if DEBUG_POSEGRAPH
    //for debug
    map<int, cv::Mat> kf_images;
#endif
};

SMART_PTR(SimplePoseGraph)
