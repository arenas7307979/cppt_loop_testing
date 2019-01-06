#include "projection_factor.h"

ProjectionFactor::ProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector2d& pt_)
    : mpCamera(camera), pt(pt_) {}

bool ProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> x3Dw(parameters_raw[1]);

    Eigen::Map<Eigen::Vector2d> residual(residual_raw);
    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;

    Eigen::Vector2d uv;
    mpCamera->Project2(x3Dc, uv);
    residual = uv - pt;

    if(jacobian_raw) {
        Eigen::Matrix<double, 2, 3> Jpi_Xc;
        Jpi_Xc = mpCamera->J2(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose;
            JXc_pose << -Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(x3Dc);
            Jpose.leftCols(6) = Jpi_Xc * JXc_pose;
            Jpose.rightCols(1).setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jpoint(jacobian_raw[1]);
            Jpoint = Jpi_Xc;
        }
    }

    return true;
}

StereoProjectionFactor::StereoProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector3d& pt_)
    : mpCamera(camera), pt(pt_) {}

bool StereoProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> x3Dw(parameters_raw[1]);

    Eigen::Map<Eigen::Vector3d> residual(residual_raw);
    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;

    Eigen::Vector3d uv_ur;
    mpCamera->Project3(x3Dc, uv_ur);
    residual = uv_ur - pt;

    if(jacobian_raw) {
        Eigen::Matrix3d Jpi_Xc;
        Jpi_Xc = mpCamera->J3(x3Dc);

        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose;
            JXc_pose << -Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(x3Dc);
            Jpose.leftCols(6) = Jpi_Xc * JXc_pose;
            Jpose.rightCols(1).setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpoint(jacobian_raw[1]);
            Jpoint = Jpi_Xc;
        }
    }

    return true;
}

namespace unary {

ProjectionFactor::ProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector2d& pt_,
                 const Eigen::Vector3d& x3Dw_)
    : camera(camera_), pt(pt_), x3Dw(x3Dw_)
{}

bool ProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                                double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<Eigen::Vector2d> residual(residual_raw);

    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;
    Eigen::Vector2d uv;
    camera->Project2(x3Dc, uv);
    residual = uv - pt;

    if(jacobian_raw) {
        Eigen::Matrix<double, 2, 3> Jpi_x3Dc;
        Jpi_x3Dc = camera->J2(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpi_pose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> Jx3Dc_pose;
            Jx3Dc_pose << -Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(x3Dc);
            Jpi_pose.leftCols(6) = Jpi_x3Dc * Jx3Dc_pose;
            Jpi_pose.rightCols(1).setZero();
        }
    }
    return true;
}

StereoProjectionFactor::StereoProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector3d& pt_,
                                               const Eigen::Vector3d& x3Dw_)
    : camera(camera_), pt(pt_), x3Dw(x3Dw_)
{}

bool StereoProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<Eigen::Vector3d> residual(residual_raw);

    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;
    Eigen::Vector3d uv_ur;
    camera->Project3(x3Dc, uv_ur);
    residual = uv_ur - pt;

    if(jacobian_raw) {
        Eigen::Matrix3d Jpi_x3Dc;
        Jpi_x3Dc = camera->J3(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpi_pose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> Jx3Dc_pose;
            Jx3Dc_pose << -Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(x3Dc);
            Jpi_pose.leftCols(6) = Jpi_x3Dc * Twc.inverse().rotationMatrix() * Jx3Dc_pose;
            Jpi_pose.rightCols(1).setZero();
        }
    }
    return true;
}

}
