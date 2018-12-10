#pragma once
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "basic_datatype/frame.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include "basic_datatype/util_datatype.h"
#include <functional>
#include "../ceres/marginalization_factor.h"
#include "../pose_graph/simple_pose_graph.h"
class SimpleBackEnd {
public:
    enum BackEndState {
        INIT,
        NON_LINEAR
    };

    SimpleBackEnd(const SimpleStereoCamPtr& camera,
                  const SlidingWindowPtr& sliding_window);
    ~SimpleBackEnd();

    void Process();
    void AddKeyFrame(const FramePtr& keyframe);
    void SetDebugCallback(const std::function<void(const std::vector<Sophus::SE3d>&,
                                                   const VecVector3d&)>& callback);
    void SetPoseGraphCallback(const std::function<void(const FramePtr keyframe)>& PG_callback);
    BackEndState mState;

private:
    bool InitSystem(const FramePtr& keyframe);
    void CreateMapPointFromStereoMatching(const FramePtr& keyframe);
    void CreateMapPointFromMotionTracking(const FramePtr& keyframe);
    void ShowResultGUI() const;
    void PubFrameToPoseGraph(const FramePtr& keyframe);
    void SlidingWindowBA(const FramePtr& new_keyframe);

    SimpleStereoCamPtr mpCamera;
    SlidingWindowPtr mpSlidingWindow;

    // buffer
    std::vector<FramePtr> mKFBuffer;
    std::mutex mKFBufferMutex;
    std::condition_variable mKFBufferCV;

    // callback function
    std::function<void(const std::vector<Sophus::SE3d>&,
                       const VecVector3d&)> mDebugCallback;

   std::function<void(const FramePtr keyframe)> mPoseGraphCallback;
};

SMART_PTR(SimpleBackEnd)
