#include "pose_graph_optimization/pose_graph_optimization.hpp"

namespace PGO {
PoseGraphOptimization::PoseGraphOptimization(const rclcpp::NodeOptions &options)
    : Node("pose_graph_optimization", options) {
  RCLCPP_INFO(this->get_logger(), "===== Pose Graph Optimization =====");

  iter_ = 0;
  linearSolver_ = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
  blockSolver_ = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver_));
  algorithm_ = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver_));

  optimizer_ = new g2o::SparseOptimizer;
  optimizer_->setAlgorithm(algorithm_);
  optimizer_->setVerbose(true);

  markerPublisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("marker", 10);

  initialData();
  initialOptimization();

  auto interval = std::chrono::duration<double>(1.0);

  this->robotPosePublish();
  this->landmarkPublish();
  this->edgePublish();

  timer_ = this->create_wall_timer(interval, [this]() -> void {
    if (iter_ < 10) {
      optimizer_->optimize(1);
      this->robotPosePublish();
      this->landmarkPublish();
      this->edgePublish();
    }
    iter_ += 1;
  });
}

void PoseGraphOptimization::initialData() {
  Eigen::Isometry2d robotPose = Eigen::Isometry2d::Identity();
  robotPoses_.emplace_back(1, robotPose);

  std::vector<std::vector<double>> landmarks;
  int32_t landmarkIndex = 1000;

  landmarks_.emplace_back(landmarkIndex + 1, Eigen::Vector2d(1.0, 0.0));
  landmarks_.emplace_back(landmarkIndex + 2, Eigen::Vector2d(1.0, 0.2));
  landmarks_.emplace_back(landmarkIndex + 3, Eigen::Vector2d(1.0, 0.4));
  landmarks_.emplace_back(landmarkIndex + 4, Eigen::Vector2d(1.0, 0.6));

  int32_t edgeIndex = 0;
  observations_.emplace_back(edgeIndex, 1, landmarkIndex + 1, Eigen::Vector2d(0.51, 0.01));
  observations_.emplace_back(++edgeIndex, 1, landmarkIndex + 2, Eigen::Vector2d(0.48, 0.18));
  observations_.emplace_back(++edgeIndex, 1, landmarkIndex + 3, Eigen::Vector2d(0.47, 0.45));
  observations_.emplace_back(++edgeIndex, 1, landmarkIndex + 3, Eigen::Vector2d(0.53, 0.57));
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

  std::for_each(observations_.begin(), observations_.end(), [this](const EdgeT &observation) {
    std::cout << observation.data << std::endl;
    g2o::EdgeSE2PointXY *observationEdge = new g2o::EdgeSE2PointXY();

    observationEdge->setId(observation.id);
    observationEdge->setVertex(0, optimizer_->vertex(observation.formId));
    observationEdge->setVertex(1, optimizer_->vertex(observation.toId));
    observationEdge->setMeasurement(observation.data);
    observationEdge->setInformation(Eigen::Matrix2d::Identity());

    optimizer_->addEdge(observationEdge);
  });

  optimizer_->initializeOptimization();
}

void PoseGraphOptimization::robotPosePublish() {
  visualization_msgs::msg::MarkerArray markerArray;

  std::for_each(robotPoses_.begin(), robotPoses_.end(), [&markerArray, this](const RobotPoseT &pose) {
    if (auto poseVertex = dynamic_cast<g2o::VertexSE2 *>(optimizer_->vertex(pose.id))) {
      auto vecPose = poseVertex->estimate().toVector();
      Eigen::Quaterniond quat;
      quat = Eigen::AngleAxisd(vecPose(2), Eigen::Vector3d::UnitZ());

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = this->get_clock()->now();
      marker.ns = "robot";
      marker.id = pose.id;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = vecPose(0);
      marker.pose.position.y = vecPose(1);
      marker.pose.position.z = 0.0;
      marker.pose.orientation.x = quat.x();
      marker.pose.orientation.y = quat.y();
      marker.pose.orientation.z = quat.z();
      marker.pose.orientation.w = quat.w();
      marker.scale.x = 0.2;
      marker.scale.y = 0.2;
      marker.scale.z = 0.2;

      marker.color.a = 1.0;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;

      markerArray.markers.push_back(marker);
    }
  });

  markerPublisher_->publish(markerArray);
}
void PoseGraphOptimization::landmarkPublish() {
  visualization_msgs::msg::MarkerArray markerArray;

  std::for_each(landmarks_.begin(), landmarks_.end(), [&markerArray, this](const LandmarkT &landmark) {
    if (auto landmarkVertex = dynamic_cast<g2o::VertexPointXY *>(optimizer_->vertex(landmark.id))) {
      auto vecPose = landmarkVertex->estimate();

      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = this->get_clock()->now();
      marker.ns = "landmark";
      marker.id = landmark.id;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = vecPose(0);
      marker.pose.position.y = vecPose(1);
      marker.pose.position.z = 0.0;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;

      marker.color.a = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;

      markerArray.markers.push_back(marker);
    }
  });

  markerPublisher_->publish(markerArray);
}

void PoseGraphOptimization::edgePublish() {
  visualization_msgs::msg::MarkerArray markerArray;

  std::for_each(observations_.begin(), observations_.end(), [&markerArray, this](const EdgeT &observation) {
    geometry_msgs::msg::Point start, end;
    if (auto poseVertex = dynamic_cast<g2o::VertexSE2 *>(optimizer_->vertex(observation.formId))) {
      auto robotPose = poseVertex->estimate().toVector();
      start.x = robotPose(0);
      start.y = robotPose(1);
      start.z = 0.0;

      auto x = observation.data(0);
      auto y = observation.data(1);

      end.x = robotPose(0) + x * std::cos(robotPose(2)) - y * std::sin(robotPose(2));
      end.y = robotPose(1) + x * std::sin(robotPose(2)) + y * std::cos(robotPose(2));
      end.z = 0.0;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "observation";
    marker.id = observation.id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.points.push_back(start);
    marker.points.push_back(end);

    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.05;

    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    markerArray.markers.push_back(marker);

    visualization_msgs::msg::Marker cubeMarker;
    cubeMarker.header.frame_id = "map";
    cubeMarker.header.stamp = this->get_clock()->now();
    cubeMarker.ns = "observation_cube";
    cubeMarker.id = observation.id;
    cubeMarker.type = visualization_msgs::msg::Marker::CUBE;
    cubeMarker.action = visualization_msgs::msg::Marker::ADD;
    cubeMarker.pose.position.x = end.x;
    cubeMarker.pose.position.y = end.y;
    cubeMarker.pose.position.z = 0;
    cubeMarker.pose.orientation.x = 0.0;
    cubeMarker.pose.orientation.y = 0.0;
    cubeMarker.pose.orientation.z = 0.0;
    cubeMarker.pose.orientation.w = 1.0;
    cubeMarker.scale.x = 0.1;
    cubeMarker.scale.y = 0.1;
    cubeMarker.scale.z = 0.1;

    cubeMarker.color.a = 0.3;
    cubeMarker.color.r = 0.0;
    cubeMarker.color.g = 0.0;
    cubeMarker.color.b = 1.0;

    markerArray.markers.push_back(cubeMarker);
  });

  markerPublisher_->publish(markerArray);
}

PoseGraphOptimization::~PoseGraphOptimization() {
  delete algorithm_;
  delete optimizer_;
}
} // namespace PGO