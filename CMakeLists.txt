cmake_minimum_required(VERSION 3.19)

project(rmvgemm VERSION 0.0.1 LANGUAGES C CXX)

set(GTEST_GIT_REPO_PATH "${CMAKE_SOURCE_DIR}/3rd-party/gtest" CACHE STRING "Path for GTest git repository")

set(CMAKE_VERBOSE_MAKEFILE ON)

option(ENABLE_TEST "Build test" ON)
option(BUILD_STATIC "Build static" ON)

set(TARGET_ARCH "RV64GVM" CACHE STRING "Target architecture")
set(BUILD_TYPE "Release" CACHE STRING "Build type")

include(cmake/common_configuration.cmake)

#
# Build library
#
add_subdirectory("${CMAKE_SOURCE_DIR}/lib")

#
# Build tests
#
if(ENABLE_TEST)
    add_subdirectory("${GTEST_GIT_REPO_PATH}")
    add_subdirectory("${CMAKE_SOURCE_DIR}/test")
endif()
