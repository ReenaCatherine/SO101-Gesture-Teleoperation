cmake_minimum_required(VERSION 3.8)
project(lerobot_mujoco)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/build_scene.py
    scripts/pose_loop_mujoco.py
    scripts/view_so101.py
    scripts/mirror_real_arm.py
    scripts/gesture_receiver_test.py
    scripts/gesture_mujoco.py
    scripts/ik_mujoco_test.py
    scripts/ik_test.py
    scripts/j6_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY urdf meshes config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
