#include "simple_pose_graph.h"
#include "ceres/local_parameterization_se3.h"
#include "utility.h"
#include <ros/ros.h>
#include "tracer.h"
#include <chrono>
#include <thread>
#define  MIN_PNPRANSAC_NUM 25
#define  MIN_USELOOPDETECT_NUM 100

void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(std::vector<cv::Point3f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

SimplePoseGraph::SimplePoseGraph(const SimpleStereoCamPtr& camera_): mpCamera(camera_)
{
    ScopedTrace st("SimplePoseGraph");
    mPoseGraphCallback = std::bind(&SimplePoseGraph::AddKeyFrameAfterBA, this,
                                   std::placeholders::_1);

    //TODO:: testing voc_path
    std::string voc_path = "/catkin_ws/src/libcppt/modules/Vocabulary/ORBvoc.bin";
    bool isloadDone = LoadORBVocabulary(voc_path);
    ORBdatabase.setVocabulary(*ORBVocabulary, false, 0);
    earliest_loop_index = -1;
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
                KeyframePtr cur_kf = std::make_shared<Keyframe>(frame, umax, db_index);
                loop_index = LoopDetection(cur_kf, db_index);

                //already find loop
                if (loop_index != -1 && loop_index != db_index)
                {
                    std::unique_lock<std::mutex> lock(keyframesMutex);
                    auto it = keyframes.find(loop_index);
                    KeyframePtr old_kf = it->second;
                    keyframesMutex.unlock();

                    //estimate relative pose between old_kf and cur_kf
                    //by PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old)
                    if(FindKFConnect(cur_kf, old_kf)){
                        //earliest_loop_index for finding start vetex in 4Dof/6Dof optimize
                        if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                            earliest_loop_index = loop_index;


#if 0
                        //for reloc and using old path.
                        //TODO:: Not finish yet.

                        Sophus::SE3d Twc_old_kf = old_kf->mTwc;
                        Sophus::SE3d Twc_cur_kf = cur_kf->mTwc;

                        //get relative info
                        Sophus::SE3d relative_T;
                        bool has_loop;
                        double relative_yaw;
                        int loop_index;
                        cur_kf->getRelativeInfo(relative_T, has_loop, relative_yaw, loop_index);

                        //Estimate new Curr Pose Twcur_new
                        Sophus::SE3d Twcur_new = Twc_old_kf * relative_T;

                        //Estimate shift between Twc_new and Twc_cur in world coordinate (global).
                        //6 DOF, From pose of loop_cur_new(Twcur_new)<- cur
                        //TODO:: IMU change to 4 DOF
                        Sophus::SE3d Shift_T = Twcur_new *  Twc_cur_kf.inverse();
#endif
                        m_optimize_buf.lock();
                        optimize_buf.push_back(cur_kf->indexInLoop);
                        m_optimize_buf.unlock();
                        mOptimizeBuffer.notify_one();
                    }
                }
                //for update map of keyframes
                std::unique_lock<std::mutex> lock(keyframesMutex);
                keyframes.insert(std::pair<int, KeyframePtr>(db_index, cur_kf)); //add curr frame to keyframes
                last_t = frame->mTwc.translation();
                db_index++;
                //publish() not finish yet
                keyframesMutex.unlock();
            }
        }
    }
}



void SimplePoseGraph::optimize6DoF(){
    //first query; then add this frame into database!
    while(1) {
        int cur_index = -1;
        int first_looped_index = -1;
        //save frame from the Back-End when pose graph slow than back-end
        std::vector<int> optimizeBuffer;
        std::unique_lock<std::mutex> lock(m_optimize_buf);
        mOptimizeBuffer.wait(lock, [&]{
            optimizeBuffer = optimize_buf;
            optimize_buf.clear();
            return !optimizeBuffer.empty();
        });
        lock.unlock();
        int i=0;

        //index is curr keyframe, it already find loop-closure.
        cur_index = optimizeBuffer[optimizeBuffer.size()-1];
        first_looped_index = earliest_loop_index;
        if (cur_index != -1)
        {
            ScopedTrace st("optimize6DoF");

            //-------start.optimize6DoF
            std::unique_lock<std::mutex> lock_kfs(keyframesMutex);
            auto cur_ptr = keyframes.find(cur_index);
            KeyframePtr cur_kf = cur_ptr->second;

            //ceres optimizer
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            ceres::LocalParameterization *pose_vertex = Sophus::AutoDiff::VertexSE3::Create();

            //set vertex
            for (std::map<int,KeyframePtr>::iterator
                 it=keyframes.begin(); it!=keyframes.end(); ++it){
                //older than first_looped_index
                if ((it)->first < first_looped_index)
                    continue;

                //Add vertex
                it->second->local_index = it->first;
                std::memcpy(it->second->vertex_data, it->second->mTwc.data(), sizeof(double) * 7);
                problem.AddParameterBlock(it->second->vertex_data, 7 , pose_vertex);

                //set constant start keyframe
                if((it)->first == first_looped_index){
                    problem.SetParameterBlockConstant(it->second->vertex_data);
                }


                //Add edge of pose that previous 5keyframe as constriant.
                std::cout << "(it->first)=" << (it->first) <<std::endl;
                std::cout << "(it->first)=" << keyframes[it->first]->mTwc.matrix() <<std::endl;
                for (int j = 1; j < 5; j++)
                {
                    if ((it->first) - j >= 0)
                    {
                        Sophus::SE3d relativePose =(keyframes[it->first-j]->mTwc.inverse() * keyframes[it->first]->mTwc);
                        std::cout << "angleX()=" << relativePose.angleX() <<std::endl;
                        std::cout << "angley()=" << relativePose.angleY() <<std::endl;
                        std::cout << "anglez()=" << relativePose.angleZ() <<std::endl;
                        if(relativePose.angleX()>10){
                        //Estimate (Twi-j).inverse * (Twi)
                        ceres::CostFunction* cost_function =
                                SixDOFError::Create(keyframes[it->first-j]->mTwc.inverse() * keyframes[it->first]->mTwc);
                        //input parameter Ti-j, Ti
                        problem.AddResidualBlock(cost_function, NULL,
                                                 keyframes[it->first-j]->vertex_data, keyframes[it->first]->vertex_data);
                        }
                    }
                }

                //Add loop detection edge, if this have been find loop-detection
                if(it->second->has_loop){
                    assert((it)->first >= first_looped_index);
                    Sophus::SE3d Match_relative_T;
                    double relative_yaw;
                    int connected_index;
                    it->second->getRelativeInfo(Match_relative_T, relative_yaw, connected_index);
                    ceres::CostFunction* cost_function = SixDOFError::Create(Match_relative_T);
                    problem.AddResidualBlock(cost_function, loss_function,
                                             keyframes[connected_index]->vertex_data, keyframes[it->first]->vertex_data);
                }
            }
            //-------end.optimize6DoF
            lock_kfs.unlock();
            ceres::Solve(options, &problem, &summary);

            std::cout << summary.FullReport() <<std::endl;

            //lock_kfs.lock();
            //-------start.UpdatePose and estimate drift Pose

            //-------end.UpdatePose and estimate drift Pose
            //lock_kfs.unlock();
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}


bool SimplePoseGraph::LoadORBVocabulary(std::string& voc_path){
    ORBVocabulary = new OrbVocabulary();
    bool bVocLoad = ORBVocabulary->loadFromBinaryFile(voc_path);
    //bool bVocLoad = ORBVocabulary->loadFromTextFile(voc_path);
    return bVocLoad;
}


int SimplePoseGraph::LoopDetection(const KeyframePtr& keyframe, const int& db_id){
    ScopedTrace st("LoopDetection");
    std::vector<cv::Mat> desc = keyframe->descriptors;
    DBoW2::QueryResults ret;
    if(db_id <= MIN_USELOOPDETECT_NUM){
        ORBdatabase.add(desc);
    }
    else{
        Tracer::TraceBegin("query&add");
        ORBdatabase.query(desc, ret, 3, db_id-MIN_USELOOPDETECT_NUM);
        ORBdatabase.add(desc);
        Tracer::TraceEnd();
    }
    bool find_loop = false;

#if DEBUG_POSEGRAPH
    //LoopDebug : 1 is show loop-detection results
    int LoopDebug = 0;
    cv::Mat compressed_image, loop_result;
    cv::Mat merage;
    if(LoopDebug){
        compressed_image = keyframe->mImgL.clone();
        int feature_num = keyframe->k_pts.size();
        cv::resize(compressed_image, compressed_image, cv::Size(376, 240));
        if (ret.size() > 0){
            putText(compressed_image, "feature_num:" + to_string(feature_num)+ " index:" + to_string(db_id), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255),1);
            //putText(compressed_image, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, MIN_USELOOPDETECT_NUM), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
        }
        kf_images[db_id] = compressed_image;
    }
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
    if (find_loop && db_id > MIN_USELOOPDETECT_NUM)
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
        if(LoopDebug){
            int tmp_index = min_index;
            auto it = kf_images.find(tmp_index);
            cv::Mat tmp_image = (it->second.clone());
            cv::resize(tmp_image, tmp_image, cv::Size(376, 240));
            putText(tmp_image, "loop score:" + to_string(ret[min_index].Score)+ " index:" + to_string(tmp_index), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255),1);
            cv::hconcat(compressed_image, tmp_image, loop_result);
            //cv::imshow("loop_result", loop_result);
            //cv::waitKey(3);
            cv::imwrite("/catkin_ws/src/libcppt/modules/Vocabulary/" + to_string(keyframe->mKeyFrameID) + ".jpg", merage);
        }
#endif
        return min_index;
    }
    else
        return -1;
}

bool SimplePoseGraph::FindKFConnect(KeyframePtr& cur_kf, KeyframePtr& old_kf){
    ScopedTrace st("FindKFConnect");
    //matching between cur_kf and old_kf by orb
    std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    std::vector<cv::Point3f> matched_x3Dw;
    matched_2d_cur = cur_kf->mv_uv;
    MatchKeyFrame(cur_kf, old_kf, matched_2d_cur, matched_2d_old, matched_x3Dw);

    //Estimate pose by PNPRansac
    std::vector<uchar> status;
    if ((int)matched_2d_cur.size() > MIN_PNPRANSAC_NUM)
    {
        status.clear();
        Eigen::Vector3d PnP_T_old;
        Eigen::Matrix3d PnP_R_old;
        Eigen::Vector3d relative_t;
        double f = mpCamera->f;
        double cx = mpCamera->cx;
        double cy = mpCamera->cy;
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
                     f, 0, cx,
                     0, f,  cy,
                     0, 0,  0);

        cv::Mat rvec, tvec, tmp_r;
        cv::Mat inliers;

        //<Assign inital pose (Tcw_cur)>
        Sophus::SE3d Tcw_cur = cur_kf->mTwc.inverse();
        cv::eigen2cv(Tcw_cur.rotationMatrix(), tmp_r);
        cv::eigen2cv(Tcw_cur.translation(), tvec);
        cv::Rodrigues(tmp_r, rvec);

        //<estimate the matched_2d_old project to plane of K=I, matched_2d_old_norm>
        //std::vector<cv::Point2f> matched_2d_old_norm;
        //for(int i=0; i<matched_2d_old.size(); i++){
        //Eigen::Vector3d normal_plane_uv;
        //mpCamera->BackProject(Eigen::Vector2d(matched_2d_old[i].x, matched_2d_old[i].y), normal_plane_uv);
        //matched_2d_old_norm.push_back(cv::Point2f(normal_plane_uv.x(), normal_plane_uv.y()));
        //}

        //<Tcw :: rvec / tvec>
        cv::solvePnPRansac(matched_x3Dw, matched_2d_old, K, cv::noArray(), rvec, tvec, true,
                           100, 8.0, 0.99, inliers);

        //cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
        //cv::solvePnPRansac(matched_x3Dw, matched_2d_old_norm, K, cv::noArray(), rvec, tvec,  true, 100, 10.0 / 460.0, 0.99, inliers);

        for (int i = 0; i < (int)matched_2d_old.size(); i++)
            status.push_back(0);

        for( int i = 0; i < inliers.rows; i++)
        {
            int n = inliers.at<int>(i);
            status[n] = 1;
        }

        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_x3Dw, status);
#if DEBUG_POSEGRAPH
        if(1){
            if(matched_2d_old.size()>MIN_PNPRANSAC_NUM){
                cv::Mat tmp_old, tmp_cur, merage;
                tmp_cur = cur_kf->mImgL.clone();
                tmp_old = old_kf->mImgL.clone();
                cv::cvtColor(tmp_cur, tmp_cur,CV_GRAY2BGR);
                cv::cvtColor(tmp_old, tmp_old,CV_GRAY2BGR);
                putText(tmp_cur,  " index:" + to_string(cur_kf->indexInLoop), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255),2);
                putText(tmp_old,  " index:" + to_string(old_kf->indexInLoop)+ " number:" + to_string(matched_2d_old.size()), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255),2);
                //for(int i=0; i<matched_2d_cur.size(); i++){
                //cv::circle(tmp_cur, matched_2d_cur[i], 3, cv::Scalar(0,0,255), -1);
                //cv::circle(tmp_old, matched_2d_old[i], 3, cv::Scalar(255,0,0), -1);
                //}
                //cv::resize(tmp_cur, tmp_cur, cv::Size(376, 240));
                //cv::resize(tmp_old, tmp_old, cv::Size(376, 240))

                cv::hconcat(tmp_cur, tmp_old, merage);

                for(int i=0; i<matched_2d_cur.size(); i++){
                    cv::circle(tmp_cur, matched_2d_cur[i], 3, cv::Scalar(0,0,255), -1);
                }

                for(int i = 0; i< (int)matched_2d_old.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i];
                    old_pt.x += (tmp_cur.cols);
                    cv::circle(merage, old_pt, 5, cv::Scalar(0, 255, 0));
                }
                for (int i = 0; i< (int)matched_2d_cur.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i];
                    old_pt.x +=  (tmp_cur.cols );
                    cv::line(merage, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
                }

                cv::imshow("merage", merage);
                cv::waitKey(3);
                cv::imwrite("/catkin_ws/src/libcppt/modules/Vocabulary/" + to_string(old_kf->mKeyFrameID) + ".jpg", merage);
            }
        }
#endif
        if(matched_2d_old.size()>MIN_PNPRANSAC_NUM){
            cv::Mat R_;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            cv::Rodrigues(rvec, R_);
            cv::cv2eigen(R_, R);
            cv::cv2eigen(tvec, t);

            //TODO:: R may not a orthogonal ?
            Sophus::SE3d Tcw_cur(R,t);

            //set loop info to cur_kf
            double relative_yaw = Utility::normalizeAngle(Utility::R2ypr(cur_kf->mTwc.rotationMatrix()).x() - Utility::R2ypr(Tcw_cur.inverse().rotationMatrix()).x());
            bool has_loop = true;
            int match_index = old_kf->indexInLoop;
            Sophus::SE3d relative_T = Tcw_cur.inverse() * cur_kf->mTwc;
            cur_kf->setRelativeInfo(relative_T, has_loop, relative_yaw, match_index);
            return true;

            //TODO:: if using IMU
            //if (abs(cur_kf->relative_yaw ) < 30.0 && cur_kf->relative_T.translation().norm() < 20.0)
            //{
            //     cur_kf->has_loop = true;
            //     cur_kf->loop_index = old_kf->loop_index;
            // return true;
            //}
        }
    }
    return  false;
}



void SimplePoseGraph::MatchKeyFrame(KeyframePtr& cur_kf, KeyframePtr& old_kf,
                                    std::vector<cv::Point2f> &matched_2d_cur,
                                    std::vector<cv::Point2f> &matched_2d_old,
                                    std::vector<cv::Point3f> &matched_x3Dw){
    ScopedTrace st("MatchKeyFrame");
    std::vector<uchar> status;
#if 1
    //mbCheckOrientation :1 use rotHist to remove outlier of not primary angle.
    int mbCheckOrientation = 1;
    int HISTO_LENGTH = 20;
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;
    //get 2d correspondence point of old_kf.
    for(int i = 0; i < (int)cur_kf->x3Dws.size(); i++)
    {
        matched_x3Dw.push_back(cv::Point3f(cur_kf->x3Dws[i].x(),
                                           cur_kf->x3Dws[i].y(),
                                           cur_kf->x3Dws[i].z()));
        cv::Point2f pt(0.f, 0.f);
        int idx;
        if (SearchInAera(cur_kf->descriptors[i], old_kf->descriptors, old_kf->k_pts, pt, idx)){
            status.push_back(1);
            if (mbCheckOrientation) {
                float rot = cur_kf->k_pts[i].angle - old_kf->k_pts[idx].angle;
                if (rot < 0.0)
                    rot += 360.0f;
                int bin = round(rot * factor);
                if (bin == HISTO_LENGTH)
                    bin = 0;
                assert(bin >= 0 && bin < HISTO_LENGTH);
                rotHist[bin].push_back(i);
            }

        }
        else
            status.push_back(0);

        matched_2d_old.push_back(pt);
    }
    if (mbCheckOrientation) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                matched_2d_old[rotHist[i][j]] = cv::Point2f(0.f, 0.f);
                status[rotHist[i][j]] = 0;
            }
        }
    }

#else
    //K-Means and BruteForce
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    cv::Mat old_kf_desc = cv::Mat::zeros((int)old_kf->k_pts.size(), 32, CV_8UC1);
    cv::Mat cur_kf_desc = cv::Mat::zeros((int)cur_kf->mv_uv.size(), 32, CV_8UC1);



    for(int i=0; i<old_kf->k_pts.size(); i++){
        for(int j=0; j<32; ++j){
            old_kf_desc.at<uchar>(i, j) = old_kf->descriptors[i].at<uchar>(0, j);
        }
    }
    for(int i=0; i<cur_kf->mv_uv.size(); i++){
        for(int j=0; j<32; ++j){
            cur_kf_desc.at<uchar>(i, j) = cur_kf->descriptors[i].at<uchar>(0, j);
        }
        matched_x3Dw.push_back(cv::Point3f(cur_kf->x3Dws[i].x(),
                                           cur_kf->x3Dws[i].y(),
                                           cur_kf->x3Dws[i].z()));
    }

    std::vector< std::vector<cv::DMatch>> matches;

    //BFMatcher matcher ( NORM_HAMMING );
    matcher->knnMatch(cur_kf_desc, old_kf_desc, matches, 2);

    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < 80) {
            status.push_back(1);
            matched_2d_old.push_back(old_kf->k_pts[matches[i][0].trainIdx].pt);
            //matched1.push_back(first_kp[matches[i][0].queryIdx]);
            //matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
        else{
            matched_2d_old.push_back( cv::Point2f(0.f, 0.f));
            status.push_back(0);
        }

    }
#endif
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_x3Dw, status);
}


void SimplePoseGraph::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++) {
        const int s = histo[i].size();
        if (s > max1) {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if (s > max2) {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        } else if (s > max3) {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float) max1) {
        ind2 = -1;
        ind3 = -1;
    } else if (max3 < 0.1f * (float) max1) {
        ind3 = -1;
    }
}


bool SimplePoseGraph::SearchInAera(const cv::Mat &window_descriptor_cur,
                                   const std::vector<cv::Mat> &descriptors_old,
                                   const std::vector<cv::KeyPoint> &keypoints_old,
                                   cv::Point2f &best_match_old, int &idx){
    cv::Point2f best_pt;
    int ORBbestDist = 100;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {
        int dis = DescriptorDistance(window_descriptor_cur, descriptors_old[i]);
        if(dis < ORBbestDist)
        {
            ORBbestDist = dis;
            bestIndex = i;
        }
    }

    if (bestIndex != -1)
    {
        best_match_old = keypoints_old[bestIndex].pt;
        idx = bestIndex;
        return true;
    }
    else{
        idx = bestIndex;
        return false;
    }
}

int SimplePoseGraph::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

