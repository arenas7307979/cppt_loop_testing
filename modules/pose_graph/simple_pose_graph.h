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
    void AddKeyFrameAfterBA(const KeyframePtr& keyframe);

    void SetPoseGraphCallback(const std::function<void(const KeyframePtr)>& PG_callback);

    //Set debugCallback
    void SetPoseDebugCallback(const std::function<void(const std::vector<KeyframePtr>&, const int& twc_type)>& callback);

    std::function<void(const KeyframePtr)> mPoseGraphCallback;

    //Pose graph optimizer
    void optimize6DoF();

    PoseGraphyState mState;

private:

    void ShowPoseGraphResultGUI(std::vector<KeyframePtr>& keyframes, int twc_type) const;

    //LoadORBVocabulary
    bool LoadORBVocabulary(std::string& voc_path);

    //Query DBOW database for find similar keyframe.
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

    //Pose graph optimizer may slower than loop-detection
    //that for save keyframe index.
    std::vector<int> optimize_buf;
    std::mutex m_optimize_buf;
    std::condition_variable mOptimizeBuffer;

    //Camera instric Parameter
    SimpleStereoCamPtr mpCamera;

    //check frame
    Eigen::Vector3d last_t;

    //mutex to protect pose graph process
    std::mutex pg_process;
    std::mutex mKFBufferMutex;
    std::vector<KeyframePtr> mKFBuffer;
    std::condition_variable mKFBufferPG;

    //for vocabulary
    OrbVocabulary* ORBVocabulary;
    OrbDatabase ORBdatabase;

    //save keyframes info in pose graph.
    std::vector<KeyframePtr> keyframes;
    std::mutex keyframesMutex;


    //Drift is optimize6DoF reult,
    //represent drift between original path and after loop-optimize path
    Sophus::SE3d T_drift;
    std::mutex m_drift;

    //debug for LC callback function
    std::function<void(const std::vector<KeyframePtr>&, int& twc_type)> mDebugCallback;

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
        Sophus::SE3<T> residual_T =  relativePose.inverse().template cast<T>() * new_relative_pose ;
        //Eigen::Quaternion<T> q = residual_T.unit_quaternion();
        //residual.head(3) = residual_T.translation();
        //residual.tail(3) << (q.x(), q.y(), q.z());
        residual.head(6) = residual_T.log();
        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d& relativePose)
    {
        return (new ceres::AutoDiffCostFunction<SixDOFError, 6, 7, 7>(
                    new SixDOFError(relativePose)));
    }
    Sophus::SE3d relativePose; // T12
};


//https://blog.csdn.net/HUAJUN998/article/details/76166307

class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(const Sophus::SE3d& relativePose,
                       const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : relativePose(relativePose), sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  const T* const p_b_ptr, const T* const q_b_ptr,
                  T* residuals_ptr) const {

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);

    Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.

    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();

    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;


    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);


    Eigen::Quaternion<T> delta_q =
        relativePose.unit_quaternion().template cast<T>() * q_ab_estimated.conjugate();


    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - relativePose.translation().template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(
      const  Sophus::SE3d &relativePose,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
        new PoseGraph3dErrorTerm(relativePose, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const  Sophus::SE3d relativePose;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};


//for testing 4dof pose graph
template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
    return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
    return angle_degrees + T(360.0);
  else
    return angle_degrees;
};

class AngleLocalParameterization {
 public:

  template <typename T>
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

    T y = yaw / T(180.0) * T(M_PI);
    T p = pitch / T(180.0) * T(M_PI);
    T r = roll / T(180.0) * T(M_PI);


    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
    inv_R[0] = R[0];
    inv_R[1] = R[3];
    inv_R[2] = R[6];
    inv_R[3] = R[1];
    inv_R[4] = R[4];
    inv_R[5] = R[7];
    inv_R[6] = R[2];
    inv_R[7] = R[5];
    inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
    r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
    r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
    r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError
{
    FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
                  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){}

    template <typename T>
    bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(t_x));
        residuals[1] = (t_i_ij[1] - T(t_y));
        residuals[2] = (t_i_ij[2] - T(t_z));
        residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                       const double relative_yaw, const double pitch_i, const double roll_i)
    {
      return (new ceres::AutoDiffCostFunction<
              FourDOFError, 4, 1, 3, 1, 3>(
                new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
    }

    double t_x, t_y, t_z;
    double relative_yaw, pitch_i, roll_i;

};

struct FourDOFWeightError
{
    FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
                  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){
                    weight = 1;
                  }

    template <typename T>
    bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
        residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
        residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
        residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                       const double relative_yaw, const double pitch_i, const double roll_i)
    {
      return (new ceres::AutoDiffCostFunction<
              FourDOFWeightError, 4, 1, 3, 1, 3>(
                new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
    }

    double t_x, t_y, t_z;
    double relative_yaw, pitch_i, roll_i;
    double weight;

};


SMART_PTR(SimplePoseGraph)
