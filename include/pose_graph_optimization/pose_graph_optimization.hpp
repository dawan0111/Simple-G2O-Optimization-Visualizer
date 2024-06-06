#ifndef POSE_GRAPH_POTIMIZATION_H
#define POSE_GRAPH_POTIMIZATION_H
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <Eigen/Core>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/types_slam2d.h>
#include <iostream>
#include <sophus/se2.hpp>
namespace PGO {

template <typename T> struct VertexData {
  int32_t id;
  T data;
  VertexData(int32_t id_, T data_) : id(id_), data(data_){};
};
template <typename T> struct EdgeData {
  int32_t id;
  int32_t formId;
  int32_t toId;
  T data;

  EdgeData(int32_t id_, int32_t formId_, int32_t toId_, T data_) : id(id_), formId(formId_), toId(toId_), data(data_){};
};

class PoseGraphOptimization : public rclcpp::Node {
  using RobotPoseT = VertexData<Eigen::Isometry2d>;
  using LandmarkT = VertexData<Eigen::Vector2d>;
  using EdgeT = EdgeData<Eigen::Vector2d>;

public:
  explicit PoseGraphOptimization(const rclcpp::NodeOptions &);
  ~PoseGraphOptimization();

private:
  void robotPosePublish();
  void landmarkPublish();
  void initialOptimization();
  void initialData();
  void optimization();

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr robotPosePublisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmarkPublisher_;

  std::vector<RobotPoseT> robotPoses_;
  std::vector<LandmarkT> landmarks_;
  std::vector<EdgeT> observations_;

  std::unique_ptr<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>> linearSolver_;
  std::unique_ptr<g2o::BlockSolverX> blockSolver_;
  g2o::OptimizationAlgorithmLevenberg *algorithm_;
  g2o::SparseOptimizer *optimizer_;
};
} // namespace PGO

#endif