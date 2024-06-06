cmake_minimum_required(VERSION 3.8)
project(pose_graph_optimization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(library_name sv_lib)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

add_library(${library_name}
  src/pose_graph_optimization.cpp
)
ament_target_dependencies(${library_name}
  rclcpp
  visualization_msgs
)
target_link_libraries(${library_name}
  Sophus::Sophus
  g2o_core
  g2o_stuff
  g2o_types_slam2d
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(pose_graph_optimization_node
  src/pose_graph_optimization_node.cpp
)
ament_target_dependencies(pose_graph_optimization_node
  rclcpp
)
target_link_libraries(pose_graph_optimization_node
  ${library_name}
)

install(TARGETS
  pose_graph_optimization_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
