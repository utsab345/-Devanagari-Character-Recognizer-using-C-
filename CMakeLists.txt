cmake_minimum_required(VERSION 3.16)

project(devanagari_character_recognizer VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core highgui imgproc dnn)

add_executable(devanagari_recognizer src/main.cpp)
target_compile_features(devanagari_recognizer PRIVATE cxx_std_17)
target_compile_options(devanagari_recognizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_link_libraries(devanagari_recognizer PRIVATE ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS devanagari_recognizer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CTest)
enable_testing()
