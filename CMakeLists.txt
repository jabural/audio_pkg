cmake_minimum_required(VERSION 3.8)
project(audio_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/AudioPlaybackAction.action"
  DEPENDENCIES
  std_msgs
  action_msgs
  ADD_LINTER_TESTS
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(dependencies
  rclcpp
  rclcpp_action
  std_msgs
  rclcpp_action
  action_msgs
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
)
add_executable(audio_listener src/audio_listener.cpp)
ament_target_dependencies(audio_listener ${dependencies})
target_link_libraries(audio_listener "${cpp_typesupport_target}")

add_executable(audio_playback src/audio_playback.cpp)
ament_target_dependencies(audio_playback ${dependencies})
target_link_libraries(audio_playback "${cpp_typesupport_target}")

add_executable(audio_playback_server src/audio_playback_server.cpp)
ament_target_dependencies(audio_playback_server ${dependencies})
target_link_libraries(audio_playback_server "${cpp_typesupport_target}")


add_executable(audio_client src/audio_client.cpp)
ament_target_dependencies(audio_client rclcpp rclcpp_action std_msgs)
target_link_libraries(audio_client "${cpp_typesupport_target}")

install(TARGETS
  audio_listener
  audio_playback
  audio_client
  audio_playback_server
  DESTINATION lib/${PROJECT_NAME})

# install(
#   DIRECTORY config
#   DESTINATION share/${PROJECT_NAME}
# )

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
