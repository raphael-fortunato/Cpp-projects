cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ConditionalGANs)

set(CMAKE_VERBOSE_MAKEFILE ON)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_definitions(-O0)
        add_definitions(-g3)
        add_definitions(-DDEBUG)
else(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_BUILD_TYPE Release)
        add_definitions(-O3)
        add_definitions(-g0)
        add_definitions(-DNDEBUG)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc calib3d ml REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories (${OpenCV_INCLUDE_DIR})
include_directories (${Torch_INCLUDE_DIR})

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})

add_executable(ConditionalGANs
		src/main.cpp
		src/conditional_gan.cpp
		src/conditional_dcgan.cpp
)
target_link_libraries(${CMAKE_PROJECT_NAME} stdc++fs)
target_link_libraries(ConditionalGANs "${TORCH_LIBRARIES}")
target_link_libraries (${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries (${CMAKE_PROJECT_NAME})

set_property(TARGET ConditionalGANs PROPERTY CXX_STANDARD 14)




