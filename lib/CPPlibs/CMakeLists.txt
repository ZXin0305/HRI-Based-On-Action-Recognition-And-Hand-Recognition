cmake_minimum_required(VERSION 2.6)
project(cppLib)
PROJECT(${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 11)
add_library(cppLib SHARED main.cpp)
FIND_PACKAGE(OpenCV 3 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
target_include_directories(
        cppLib PUBLIC
        "/usr/include/python3.8"
)
target_link_libraries(cppLib ${OpenCV_LIBS}) 