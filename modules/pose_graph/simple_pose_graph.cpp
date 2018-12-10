#include "simple_pose_graph.h"
#include <ros/ros.h>
#include "tracer.h"
#include <chrono>
#include <thread>

SimplePoseGraph::SimplePoseGraph(const SimpleStereoCamPtr& camera_): mpCamera(camera_)
{
    mPoseGraphCallback = std::bind(&SimplePoseGraph::AddKeyFrameAfterBA, this,
                                   std::placeholders::_1);

    //TODO:: testing voc_path
    std::string voc_path = "/catkin_ws/src/libcppt/modules/Vocabulary/ORBvoc.bin";
    bool isloadDone = LoadORBVocabulary(voc_path);
    ORBdatabase.setVocabulary(*ORBVocabulary, false, 0);

    //for check frame sequence
    last_t.x() = -100;
    last_t.y() = -100;
    last_t.z() = -100;

    //for IC angle of ORB descriptor
    umax.resize(HALF_PATCH_SIZE + 1);

    //from ORBSLAM for estiamte IC angle
    //pre-compute the end of a row in a circular patch
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;

        umax[v] = v0;
        ++v0;
    }
}

SimplePoseGraph::~SimplePoseGraph() {}


void SimplePoseGraph::AddKeyFrameAfterBA(const FramePtr& frame) {
    assert(frame->mIsKeyFrame);
    mKFBufferMutex.lock();
    mKFBuffer.push_back(frame);
    mKFBufferMutex.unlock();
    mKFBufferPG.notify_one();
}

void SimplePoseGraph::Process(){
    //first query; then add this frame into database!
    while(1) {
        //save frame from the Back-End when pose graph slow than back-end
        std::vector<FramePtr> v_frame;
        std::unique_lock<std::mutex> lock(mKFBufferMutex);
        mKFBufferPG.wait(lock, [&]{
            v_frame = mKFBuffer;
            mKFBuffer.clear();
            return !v_frame.empty();
        });
        lock.unlock();
        int i=0;

        for(auto& frame : v_frame) {
            std::unique_lock<std::mutex> lock(pg_process);
            if((frame->mTwc.translation() - last_t).norm() > 0){
            int loop_index = -1;
            KeyframePtr cur_kf = std::make_shared<Keyframe>(frame, umax);
            loop_index = LoopDetection(cur_kf, db_index);
            keyframes.insert(std::pair<int, KeyframePtr>(db_index, cur_kf));
            //find loop!!
            if (loop_index != -1)
            {
            auto it = keyframes.find(loop_index);
            KeyframePtr old_kf = it->second;
            //estimate relative pose between old_kf and cur_kf
            //by PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old)

            }
            last_t = frame->mTwc.translation();
            db_index++;
            pg_process.unlock();
            //for testing
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
            }
        }
    }
}


bool SimplePoseGraph::LoadORBVocabulary(std::string& voc_path){
    ORBVocabulary = new OrbVocabulary();
    bool bVocLoad = ORBVocabulary->loadFromBinaryFile(voc_path);
    return bVocLoad;
}


int SimplePoseGraph::LoopDetection(const KeyframePtr& keyframe, const int& db_id){

    std::vector<cv::Mat> desc = keyframe->descriptors;
    DBoW2::QueryResults ret;

    if(db_id <= 50){
        ORBdatabase.add(desc);
    }
    else{
        ORBdatabase.query(desc, ret, 4, db_id-50);
        ORBdatabase.add(desc);
    }

    bool find_loop = false;

#if DEBUG_POSEGRAPH
    cv::Mat compressed_image, loop_result;
    cv::Mat merage;
    compressed_image = keyframe->mImgL.clone();
    int feature_num = keyframe->k_pts.size();
    cv::resize(compressed_image, compressed_image, cv::Size(376, 240));
    if (ret.size() > 0){
        putText(compressed_image, "feature_num:" + to_string(feature_num)+ " index:" + to_string(db_id), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        //putText(compressed_image, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    kf_images[db_id] = compressed_image;
#endif

    // a good match with its nerghbour
    if (ret.size() >= 1 && ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
            }

        }
    if (find_loop && db_id > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            //To select the oldest matching frame
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
#if DEBUG_POSEGRAPH
        //show dbow2 query result (current keyframe in pose graph match with orb database)
        if(1){
        int tmp_index = min_index;
        auto it = kf_images.find(tmp_index);
        cv::Mat tmp_image = (it->second.clone());
        cv::resize(tmp_image, tmp_image, cv::Size(376, 240));
        putText(tmp_image, "loop score:" + to_string(ret[min_index].Score)+ " index:" + to_string(tmp_index), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        cv::hconcat(compressed_image, tmp_image, loop_result);
        cv::imshow("loop_result", loop_result);
        cv::waitKey(3);
        }
#endif
        return min_index;
    }
    else
        return -1;

}

bool SimplePoseGraph::FindKFConnect(KeyframePtr& cur_kf, KeyframePtr& old_kf){
    //get old_kf of desc, keypoint position that points have mappoint info.
    cv::Mat old_desc;
    std::vector<cv::KeyPoint> old_kps;
    old_desc = cv::Mat::zeros((int)old_kf->mvMapPoint.size(), 32, CV_8UC1);
    for(int i=0; i<old_kf->mvMapPoint.size(); i++){
        old_desc.row(i)= old_kf->descriptors[i];
        old_kps.emplace_back(old_kf->k_pts[i]);
    }

    std::vector<uchar> status;

    //matching between cur_kf and old_kf by orb
    std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    std::vector<cv::Point3f> matched_x3Dw;
    std::vector<double> matched_id;
    for(int i = 0; i < (int)cur_kf->mvMapPoint.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
//        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
//          status.push_back(1);
//        else
//          status.push_back(0);
//        matched_2d_old.push_back(pt);
//        matched_2d_old_norm.push_back(pt_norm);


        status.clear();
    }
}

