# Build file for the compliant grasp service and its gate diagnostics.
#
# The original build file was not recovered with the sources; this one was
# written for this repository and has not been built against the vendor SDK.
#
#   cmake -S . -B build -DSTARK_SDK_DIR=<path to the BrainCo Stark SDK>
#   cmake --build build -j4
#
# STARK_SDK_DIR must contain stark-sdk.h (directly or under include/) and the
# SDK shared library (directly or under lib/).
cmake_minimum_required(VERSION 3.16)
project(brainco_compliant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(STARK_SDK_DIR "" CACHE PATH "Root of the BrainCo Stark SDK")
find_path(STARK_SDK_INCLUDE stark-sdk.h
          HINTS ${STARK_SDK_DIR} ${STARK_SDK_DIR}/include REQUIRED)
# The SDK ships its shared library as libbc_stark_sdk.so.
find_library(STARK_SDK_LIB NAMES bc_stark_sdk
             HINTS ${STARK_SDK_DIR} ${STARK_SDK_DIR}/lib REQUIRED)
find_package(Threads REQUIRED)

foreach(target brainco_grasp_service h3_spherical_grasp h2_index_impedance
               h0_index_current_visible)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${STARK_SDK_INCLUDE})
  target_link_libraries(${target} PRIVATE ${STARK_SDK_LIB} Threads::Threads)
endforeach()
