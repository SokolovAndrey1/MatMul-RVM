
# Base configuration for all files in project.
# (including custom configurations in lib/CMakeLists.txt)
add_library(BaseConfiguration INTERFACE)

target_compile_options(BaseConfiguration
INTERFACE
    -Wall
    -Wextra
)

if(BUILD_STATIC)
    target_link_options(BaseConfiguration INTERFACE -static)
endif()


if("${TARGET_ARCH}" STREQUAL "RV64GC")
    target_compile_options(BaseConfiguration INTERFACE -march=rv64gcv)
    target_compile_definitions(BaseConfiguration INTERFACE -DRV64G)
elseif("${TARGET_ARCH}" STREQUAL "RV64GV")
    target_compile_options(BaseConfiguration INTERFACE -march=rv64gcv0p7)
    target_compile_definitions(BaseConfiguration INTERFACE -DRV64GV)
elseif("${TARGET_ARCH}" STREQUAL "RV64GVM")
    target_compile_options(BaseConfiguration INTERFACE -march=rv64gcv0p7_xtheadmatrix)
    target_compile_definitions(BaseConfiguration INTERFACE -DRV64GVM)
elseif("${TARGET_ARCH}" STREQUAL "X86")
    target_compile_definitions(BaseConfiguration INTERFACE -DX86)
else()
    message(FATAL_ERROR "Unsupported TARGET_ARCH = ${TARGET_ARCH}")
endif()

# Base configuration with optimization flags
# Used for all files except custom configuration in lib/CMakeLists.txt and benchmarks
add_library(BaseOptConfiguration INTERFACE)
if(BUILD_TYPE STREQUAL "Release")
    target_compile_options(BaseOptConfiguration INTERFACE -O3)
else() 
    target_compile_options(BaseOptConfiguration
    INTERFACE
        -O0
        -g
        -DNDEBUG
    )
endif()

# Common configuration.
# Used for all files except custom configuration in lib/CMakeLists.txt and benchmarks
add_library(CommonConfiguration INTERFACE)
target_link_libraries(CommonConfiguration INTERFACE BaseConfiguration BaseOptConfiguration)
