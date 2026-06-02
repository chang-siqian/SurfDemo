cmake_minimum_required(VERSION 3.10)
project(SurfDemoJetson)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs highgui
               cudastereo cudafilters cudawarping
               calib3d features2d flann)

add_executable(sgm_cuda ../SGMCuda.cpp)
target_link_libraries(sgm_cuda ${OpenCV_LIBS})

add_executable(sgm_adaptive ../SGMCudaAdaptive.cpp)
target_link_libraries(sgm_adaptive ${OpenCV_LIBS})

add_executable(kitti_eval ../KITTIEval.cpp)
target_link_libraries(kitti_eval ${OpenCV_LIBS} stdc++fs)
