#include "simple_pose_graph.h"
#include "ceres/local_parameterization_se3.h"
#include "utility.h"
#include <ros/ros.h>
#include "tracer.h"
#include <chrono>
#include <thread>
#define  MIN_PNPRANSAC_NUM 25
#define  MIN_USELOOPDETECT_NUM 50
#define  SIX_DOF 1
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
}

SimplePoseGraph::~SimplePoseGraph() {}


void SimplePoseGraph::AddKeyFrameAfterBA(const KeyframePtr& keyframe) {
    mKFBufferMutex.lock();
    mKFBuffer.push_back(keyframe);
    mKFBufferMutex.unlock();
    mKFBufferPG.notify_one();
}

void SimplePoseGraph::Process(){
    //first query; then add this frame into database!
    while(1) {
        //save frame from the Back-End when pose graph slow than back-end
        std::vector<KeyframePtr> v_frame;
        std::unique_lock<std::mutex> lock(mKFBufferMutex);
        mKFBufferPG.wait(lock, [&]{
            v_frame = mKFBuffer;
            mKFBuffer.clear();
            return !v_frame.empty();
        });
        lock.unlock();
        int i=0;

        for(auto& cur_kf : v_frame) {

            std::unique_lock<std::mutex> lock(pg_process);
            if((cur_kf->vio_mTwc.translation() - last_t).norm() > 0){
                int loop_index = -1;
                cur_kf->indexInLoop = db_index;
                //std::cout  << "T_drift=" << T_drift.matrix() << std::endl;
                //KeyframePtr cur_kf = std::make_shared<Keyframe>(frame, umax, db_index);
                loop_index = LoopDetection(cur_kf, db_index);

                //already find loop
                if (loop_index != -1 && loop_index != db_index)
                {
                    keyframesMutex.lock();
                    KeyframePtr old_kf = keyframes[loop_index];
                    keyframesMutex.unlock();

                    //estimate relative pose between old_kf and cur_kf
                    if(FindKFConnect(cur_kf, old_kf)){
                        //earliest_loop_index for finding start vetex in 4Dof/6Dof optimize
                        if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                            earliest_loop_index = loop_index;

                        m_optimize_buf.lock();
                        optimize_buf.push_back(cur_kf->indexInLoop);
                        m_optimize_buf.unlock();
                        mOptimizeBuffer.notify_one();
                    }
                }
                //for update map of keyframes
                keyframesMutex.lock();
                cur_kf->mTwc = cur_kf->vio_mTwc * T_drift;
                keyframes.push_back(cur_kf); //add curr frame to keyframes
                last_t = cur_kf->vio_mTwc.translation();
                db_index++;
                ShowPoseGraphResultGUI(keyframes, 0);
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
        m_optimize_buf.unlock();
        int i=0;

        //index is curr keyframe, it already find loop-closure.
        cur_index = optimizeBuffer[optimizeBuffer.size()-1];
        first_looped_index = earliest_loop_index;
        if (cur_index != -1 && first_looped_index != -1)
        {
            ScopedTrace st("optimize6DoF");
            //-------start.optimize6DoF
            keyframesMutex.lock();
            KeyframePtr cur_kf = keyframes[cur_index];
            //ceres optimizer
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = 100;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
#if SIX_DOF
            ceres::LocalParameterization* quaternion_local_parameterization =
                    new ceres::EigenQuaternionParameterization;
#else
            ceres::LocalParameterization* angle_local_parameterization =
                    AngleLocalParameterization::Create();
#endif

            //information matrix
            Eigen::Matrix<double, 6, 6> information;
            information.setIdentity();

            int i_n=0;
            //set vertex
            for (auto &it : keyframes){

                //older than first_looped_index
                if (it->indexInLoop < first_looped_index)
                    continue;
                //Add vertex
                //Add rotation / translation, Sophus Eigen Quaternion is qx,y,z, qw;
#if SIX_DOF
                std::memcpy(it->c_rotation, it->vio_mTwc.data(), sizeof(double) * 4);
                std::memcpy(it->c_translation, it->vio_mTwc.data()+4, sizeof(double) * 3);
                problem.AddParameterBlock(it->c_rotation, 4 , quaternion_local_parameterization);
                problem.AddParameterBlock(it->c_translation, 3);
#else
                Eigen::Quaterniond tmp_q = it->vio_mTwc.so3().unit_quaternion();
                Eigen::Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                std::memcpy(it->euler_array, euler_angle.data(), sizeof(double) * 3);
                std::memcpy(it->c_translation, it->vio_mTwc.data()+4, sizeof(double) * 3);
                it->q_array = tmp_q;
                problem.AddParameterBlock(it->euler_array, 1, angle_local_parameterization);
                problem.AddParameterBlock(it->c_translation, 3);
#endif


#if SIX_DOF
                //set constant start keyframe
                if((it)->indexInLoop == first_looped_index){
                    //problem.SetParameterBlockConstant(it->second->vertex_data);
                    problem.SetParameterBlockConstant(it->c_rotation);
                    problem.SetParameterBlockConstant(it->c_translation);
                }
#else
                if((it)->indexInLoop == first_looped_index){
                    //problem.SetParameterBlockConstant(it->second->vertex_data);
                    problem.SetParameterBlockConstant(it->euler_array);
                    problem.SetParameterBlockConstant(it->c_translation);
                }
#endif

                //Add edge of pose that previous 5keyframe as constriant.
#if SIX_DOF
                for (int j = 1; j < 5; j++)
                {
                    if ((it->indexInLoop - j) > first_looped_index)
                    {
                        Sophus::SE3d relativePose = keyframes[(it->indexInLoop) - j]->vio_mTwc.inverse() * it->vio_mTwc;
                        //Estimate (Twi-j).inverse * (Twi)
                        ceres::CostFunction* cost_function =
                                PoseGraph3dErrorTerm::Create(relativePose, information);
                        //input parameter Ti-j, Ti
                        problem.AddResidualBlock(cost_function, loss_function,
                                                 keyframes[(it->indexInLoop) - j]->c_translation,
                                keyframes[(it->indexInLoop) - j]->c_rotation,
                                it->c_translation,
                                it->c_rotation);
                        //?????, rotation
                        problem.SetParameterization(keyframes[(it->indexInLoop) - j]->c_rotation,
                                quaternion_local_parameterization);
                        problem.SetParameterization(it->c_rotation,
                                                    quaternion_local_parameterization);
                    }
                }

#else
                for (int j = 1; j < 5; j++)
                {
                    if ((it->indexInLoop) - j >= 0 && (it->indexInLoop) - j > first_looped_index)
                    {
                        Eigen::Vector3d euler_conncected = Utility::R2ypr(keyframes[(it->indexInLoop) - j]->q_array.toRotationMatrix());
                        Sophus::SE3d relativePose = keyframes[(it->indexInLoop) - j]->vio_mTwc.inverse() * it->vio_mTwc;
                        double relative_yaw = it->euler_array[0] - keyframes[(it->indexInLoop) - j]->euler_array[0];
                        //Estimate (Twi-j).inverse * (Twi)
                        ceres::CostFunction* cost_function = FourDOFError::Create( relativePose.translation().x(),
                                                                                   relativePose.translation().y(),
                                                                                   relativePose.translation().z(),
                                                                                   relative_yaw,
                                                                                   euler_conncected.y(),
                                                                                   euler_conncected.z());

                        //input parameter Ti-j, Ti
                        problem.AddResidualBlock(cost_function, NULL, keyframes[(it->indexInLoop) - j]->euler_array,
                                keyframes[(it->indexInLoop) - j]->c_translation,
                                it->euler_array,
                                it->c_translation);
                    }
                }

#endif

                //Add loop detection edge, if this have been find loop-detection
#if SIX_DOF
                if(it->has_loop){
                    assert((it)->indexInLoop >= first_looped_index);
                    Sophus::SE3d Match_relative_T;
                    double relative_yaw;
                    int connected_index;
                    it->getRelativeInfo(Match_relative_T, relative_yaw, connected_index);
                    ceres::CostFunction* cost_function = PoseGraph3dErrorTerm::Create(Match_relative_T, information);
                    problem.AddResidualBlock(cost_function, loss_function,
                                             keyframes[connected_index]->c_translation, keyframes[connected_index]->c_rotation,
                                             it->c_translation, it->c_rotation);
                    //?????, rotation
                    problem.SetParameterization(keyframes[connected_index]->c_rotation,
                                                quaternion_local_parameterization);
                    problem.SetParameterization(it->c_rotation,
                                                quaternion_local_parameterization);
                }
#else
                if(it->has_loop){
                    Sophus::SE3d Match_relative_T;
                    double relative_yaw;
                    int connected_index;
                    it->getRelativeInfo(Match_relative_T, relative_yaw, connected_index);

                    Eigen::Vector3d euler_conncected = Utility::R2ypr(keyframes[connected_index]->q_array.toRotationMatrix());
                    Eigen::Vector3d relative_t = Match_relative_T.translation();

                    ceres::CostFunction* cost_function = FourDOFWeightError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                     relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, loss_function,
                                             keyframes[connected_index]->euler_array,
                                             keyframes[connected_index]->c_translation,
                                             it->euler_array,
                                             it->c_translation);

                }
#endif
                if ((it)->indexInLoop == cur_index)
                    break;
            }
            //-------end.optimize6DoF
            keyframesMutex.unlock();
            ceres::Solve(options, &problem, &summary);

#if SIX_DOF
            //-------start.UpdatePose and estimate drift Pose
            //std::vector<KeyframePtr> keyframes_debug;
            keyframesMutex.lock();
            std::vector<Sophus::SE3d> keyframes_Twc;
            for (auto &it : keyframes){
                //older than first_looped_index
                if (it->indexInLoop < first_looped_index)
                    continue;


                //keyframes_debug.emplace_back(it);
                for (int k = 0; k < 7; k++)
                {
                    if(k<4){
                        it->vertex_data[k] = it->c_rotation[k];
                    }
                    else{
                        it->vertex_data[k] = it->c_translation[k-4];
                    }
                }
                Eigen::Map<Sophus::SE3d> cur_VIOnew_(it->vertex_data);
                Sophus::SE3d cur_VIOnew = cur_VIOnew_;
                it->mTwc = cur_VIOnew;
                //keyframes_Twc.push_back(cur_VIOnew);
                if ((it)->indexInLoop == cur_index)
                    break;
            }
            //ShowPoseGraphResultGUI(keyframes_debug, 0);
            //ShowPoseGraphResultGUI(keyframes_Twc, 0);

            //-------start. estimate drift Pose
            m_drift.lock();
            T_drift = keyframes[cur_index]->vio_mTwc.inverse() * keyframes[cur_index]->mTwc;
            //            //???from vins
            //            //T_drift.translation() = keyframes[cur_index]->mTwc.translation() -
            //            //        (T_drift.rotationMatrix() * cur_VIO.translation());
            m_drift.unlock();

            //-------start. update keyframe pose that index bigger than cur_index
            for (auto &it : keyframes){
                if ((it)->indexInLoop > cur_index)
                {
                    it->mTwc  = it->vio_mTwc * T_drift;
                }
            }
            keyframesMutex.unlock();
#else
            keyframesMutex.lock();
            for (auto &it : keyframes){
                //older than first_looped_index
                if (it->indexInLoop < first_looped_index)
                    continue;
                Eigen::Quaterniond tmp_q;
                tmp_q = Utility::ypr2R(Eigen::Vector3d(it->euler_array[0], it->euler_array[1], it->euler_array[2]));
                Eigen::Vector3d tmp_t = Eigen::Vector3d(it->c_translation[0], it->c_translation[1], it->c_translation[2]);
                Eigen::Matrix3d tmp_r = tmp_q.toRotationMatrix();
                Sophus::SE3d keyframes_Twc(tmp_r, tmp_t);
                it->mTwc = keyframes_Twc;
                if ((it)->indexInLoop == cur_index)
                    break;
            }
            keyframesMutex.unlock();


            //-------start. estimate drift Pose
            m_drift.lock();
            T_drift = cur_VIO.inverse() * keyframes[cur_index]->mTwc;
            m_drift.unlock();

            //-------start. update keyframe pose that index bigger than cur_index
            keyframesMutex.lock();
            for (auto &it : keyframes){
                if ((it)->indexInLoop > cur_index)
                    it->mTwc  = it->mTwc * T_drift;
            }
            keyframesMutex.unlock();
#endif
        }
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}


void SimplePoseGraph::SetPoseDebugCallback(const std::function<void(const std::vector<KeyframePtr>&, const int& twc_type)>& callback)
{
    mDebugCallback = callback;
}

void SimplePoseGraph::ShowPoseGraphResultGUI(std::vector<KeyframePtr>& keyframes, int twc_type) const
{
    if(!mDebugCallback)
        return;

    mDebugCallback(keyframes, twc_type);
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

    Tracer::TraceBegin("query&add");
    ORBdatabase.query(desc, ret, 3, db_id-MIN_USELOOPDETECT_NUM);
    ORBdatabase.add(desc);
    Tracer::TraceEnd();
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
    if (ret.size() >= 1 && ret[0].Score > 0.05){
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
            }

        }
    }
    if (find_loop && db_id >= MIN_USELOOPDETECT_NUM)
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
        Sophus::SE3d Tcw_cur = cur_kf->vio_mTwc.inverse();
        cv::eigen2cv(Tcw_cur.rotationMatrix(), tmp_r);
        cv::eigen2cv(Tcw_cur.translation(), tvec);
        cv::Rodrigues(tmp_r, rvec);

        //<Tcw :: rvec / tvec>
        cv::solvePnPRansac(matched_x3Dw, matched_2d_old, K, cv::noArray(), rvec, tvec, true,
                           100, 10.0, 0.99, inliers);

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
                putText(tmp_old,  " index:" + to_string(old_kf->indexInLoop)+ " number:" + to_string(matched_2d_old.size()),
                        cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255),2);
                cv::hconcat(tmp_cur, tmp_old, merage);


                for(int i = 0; i< (int)matched_2d_old.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i];
                    old_pt.x += (tmp_cur.cols);
                    cv::circle(merage, old_pt, 5, cv::Scalar(0, 255, 0));
                    cv::circle(merage, matched_2d_cur[i], 5, cv::Scalar(0, 255, 0));
                }
                for (int i = 0; i< (int)matched_2d_cur.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i];
                    old_pt.x +=  (tmp_cur.cols );
                    cv::line(merage, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
                }

                cv::imshow("merage", merage);
                cv::waitKey(4);cur_index
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
            double relative_yaw = Utility::normalizeAngle(Utility::R2ypr(cur_kf->vio_mTwc.rotationMatrix()).x() - Utility::R2ypr(Tcw_cur.inverse().rotationMatrix()).x());
            Sophus::SE3d relative_T = Tcw_cur * cur_kf->vio_mTwc;

            //TODO:: need change this condition that only for VIO(with IMU).
            if (abs(relative_yaw) < 30.0 && relative_T.translation().norm() < 20.0)
            {
                bool has_loop = true;
                int match_index = old_kf->indexInLoop;
                cur_kf->setRelativeInfo(relative_T, has_loop, relative_yaw, match_index);
                //for PnPdebug show in rviz
#if 0
                std::vector<Sophus::SE3d> PnPresult;
                PnPresult.resize(1);
                PnPresult[0] = Tcw_cur.inverse();
                ShowPoseGraphResultGUI(PnPresult);
#endif
            }
            return true;
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
    //mbCheckOrientation :1 use rotHist to remove outlier of not primary angle.
    int mbCheckOrientation = 1;
    int HISTO_LENGTH = 25;
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

