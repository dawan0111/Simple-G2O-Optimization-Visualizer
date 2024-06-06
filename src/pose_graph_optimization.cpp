#include "pose_graph_optimization/pose_graph_optimization.hpp"

namespace PGO {
PoseGraphOptimization::PoseGraphOptimization(const rclcpp::NodeOptions &options)
    : Node("pose_graph_optimization", options) {
  RCLCPP_INFO(this->get_logger(), "===== Pose Graph Optimization =====");

  linearSolver_ = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  blockSolver_ = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver_));
  auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver_));

  optimizer_ = new g2o::SparseOptimizer;
  optimizer_->setAlgorithm(algorithm);
  optimizer_->setVerbose(true);

  robotPosePublisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("robot", 10);
  landmarkPublisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("landmark", 10);

  initialData();
  initialOptimization();

  auto interval = std::chrono::duration<double>(1.0);
  timer_ = this->create_wall_timer(interval, [this]() -> void {
    this->robotPosePublish();
    this->landmarkPublish();
  });
}

void PoseGraphOptimization::initialData() {
  Eigen::Isometry2d robotPose = Eigen::Isometry2d::Identity();
  robotPoses_.emplace_back(1, std::move(robotPose));

  std::vector<std::vector<double>> landmarks;
  int32_t landmarkIndex = 1000;

  landmarks_.emplace_back(++landmarkIndex, Eigen::Vector2d(1.0, 1.0));
  landmarks_.emplace_back(++landmarkIndex, Eigen::Vector2d(1.0, 3.0));
  landmarks_.emplace_back(++landmarkIndex, Eigen::Vector2d(1.0, 5.0));

  int32_t edgeIndex = 0;
  observations_.emplace_back(++edgeIndex, 1, 1001, Eigen::Vector2d(0, 1.0));
  observations_.emplace_back(++edgeIndex, 1, 1002, Eigen::Vector2d(0, 3.0));
  observations_.emplace_back(++edgeIndex, 1, 1003, Eigen::Vector2d(0, 5.0));
}

void PoseGraphOptimization::initialOptimization() {
  std::for_each(robotPoses_.begin(), robotPoses_.end(), [this](const auto &pose) {
    g2o::VertexSE2 *robotPose = new g2o::VertexSE2();

    robotPose->setId(pose.id);
    robotPose->setEstimate(g2o::SE2(pose.data));

    optimizer_->addVertex(robotPose);
  });

  std::for_each(landmarks_.begin(), landmarks_.end(), [this](const auto &landmark) {
    g2o::VertexPointXY *landmarkVertex = new g2o::VertexPointXY();

    landmarkVertex->setId(landmark.id);
    landmarkVertex->setEstimate(landmark.data);

    optimizer_->addVertex(landmarkVertex);
  });

  std::for_each(observations_.begin(), observations_.end(), [this](const auto &observation) {
    g2o::EdgeSE2PointXY *observationEdge = new g2o::EdgeSE2PointXY();

    observationEdge->setId(observation.id);
    observationEdge->setVertex(0, optimizer_->vertex(observation.formId));
    observationEdge->setVertex(1, optimizer_->vertex(observation.toId));
    observationEdge->setMeasurement(observation.data);

    optimizer_->addEdge(observationEdge);
  });
}

void PoseGraphOptimization::robotPosePublish() {
  visualization_msgs::msg::MarkerArray markerArray;

  std::for_each(robotPoses_.begin(), robotPoses_.end(), [&markerArray, this](const auto &pose) {
    auto trans = pose.data.translation();
    auto rotation = pose.data.rotation();
    double angle = std::atan2(rotation(1, 0), rotation(0, 0));
    Eigen::Quaterniond quat;
    quat = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "sphere";
    marker.id = pose.id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = trans(0);
    marker.pose.position.y = trans(1);
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
    marker.color.a = 1.0; // Alpha must be non-zero
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    markerArray.markers.push_back(marker);
  });

  robotPosePublisher_->publish(markerArray);
}
void PoseGraphOptimization::landmarkPublish() {
  visualization_msgs::msg::MarkerArray markerArray;

  std::for_each(landmarks_.begin(), landmarks_.end(), [&markerArray, this](const auto &landmark) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "cube";
    marker.id = landmark.id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = landmark.data(0);
    marker.pose.position.y = landmark.data(1);
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
    marker.color.a = 1.0; // Alpha must be non-zero
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;

    markerArray.markers.push_back(marker);
  });

  landmarkPublisher_->publish(markerArray);
}

PoseGraphOptimization::~PoseGraphOptimization() {
  delete algorithm_;
  delete optimizer_;
}
} // namespace PGO