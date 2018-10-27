#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    Eigen::MatrixXd Ht = H.transpose();

    Eigen::VectorXd z_pred = H_ * x_;
    Eigen::VectorXd y = z - z_pred;
    Eigen::MatrixXd S = H_ * P_ * Ht + R_;
    Eigen::MatrixXd K = P * Ht * S.inverse();

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

