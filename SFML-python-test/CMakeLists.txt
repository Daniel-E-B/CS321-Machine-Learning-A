cmake_minimum_required(VERSION 3.13)
project(SFML_python_test)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp)

find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} sfml-graphics sfml-audio)
