cmake_minimum_required(VERSION 3.12)
project(opencv)
set(PROJECT_EXEC "food-detection")

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_EXEC} main.cpp)
target_link_libraries(${PROJECT_EXEC} ${OpenCV_LIBS} )