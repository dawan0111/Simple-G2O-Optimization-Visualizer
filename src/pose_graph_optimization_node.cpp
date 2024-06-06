#include "pose_graph_optimization/pose_graph_optimization.hpp"
#include "rclcpp/rclcpp.hpp"
#include <iostream>

int32_t main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  const rclcpp::NodeOptions options;

  auto poseGraphOptimizationNode = std::make_shared<PGO::PoseGraphOptimization>(options);

  rclcpp::spin(poseGraphOptimizationNode->get_node_base_interface());
  rclcpp::shutdown();

  return 0;
}