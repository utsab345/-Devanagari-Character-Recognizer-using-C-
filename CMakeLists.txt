cmake_minimum_required(VERSION 3.30)  # Use your current version
project(opencvtest VERSION 0.1.0 LANGUAGES C CXX)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR to the correct path.")
endif()

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(opencvtest main.cpp)

# Link OpenCV libraries
target_link_libraries(opencvtest ${OpenCV_LIBS})

# Enable testing
include(CTest)
enable_testing()

# CPack configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
