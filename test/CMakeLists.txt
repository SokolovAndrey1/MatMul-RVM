file(GLOB SOURCES "*/*.cpp")

function(link_libs target)
    target_include_directories(${target}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/test/include
        ${PROJECT_SOURCE_DIR}/3rd-party/gtest/googletest/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )

    target_link_libraries(${target}
    PUBLIC
        GTest::gtest
        GTest::gtest_main
        rmvgemm
    PRIVATE
        CommonConfiguration
    )
endfunction()


add_executable(test_ref_square "${CMAKE_CURRENT_SOURCE_DIR}/src/test_ref_square.cpp")
link_libs(test_ref_square)
add_executable(test_rvv_square "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rvv_square.cpp")
link_libs(test_rvv_square)
add_executable(test_rvm_square "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rvm_square.cpp")
link_libs(test_rvm_square)

add_executable(test_ref_nonsquare "${CMAKE_CURRENT_SOURCE_DIR}/src/test_ref_nonsquare.cpp")
link_libs(test_ref_nonsquare)
add_executable(test_rvv_nonsquare "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rvv_nonsquare.cpp")
link_libs(test_rvv_nonsquare)
add_executable(test_rvm_nonsquare "${CMAKE_CURRENT_SOURCE_DIR}/src/test_rvm_nonsquare.cpp")
link_libs(test_rvm_nonsquare)


