cmake_minimum_required(VERSION 2.8.8)
project(e57processor)

set (e57processor_VERSION_MAJOR 0)
set (e57processor_VERSION_MINOR 1)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

if(CMAKE_COMPILER_IS_GNUCXX)
    if (NOT WIN32)
        add_definitions(-fPIC)
    else()
        add_definitions(-DM_LOG2E=1.44269504088896340736)
    endif()
	add_definitions(-O3)
	add_definitions(-g)
	add_definitions(-std=c++1y)
    add_definitions(-Wall)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-unused-local-typedefs)
    add_definitions(-Wno-deprecated-declarations)
endif()

find_package(Eigen)
find_package(PCL COMPONENTS common io search filters keypoints features)
find_package(Boost COMPONENTS system program_options filesystem random)
find_package(LibE57)

file (GLOB_RECURSE obj RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.cpp")
if (EIGEN_FOUND AND PCL_FOUND AND Boost_FOUND AND LIBE57_FOUND)
	include_directories(${EIGEN_INCLUDE_DIRS})
	include_directories(${PCL_INCLUDE_DIRS})
	include_directories(${Boost_INCLUDE_DIRS})
	include_directories(${LIBE57_INCLUDE_DIRS})

	add_executable(e57processor ${obj})
	target_link_libraries(e57processor ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${LIBE57_LIBRARIES} "dl")

	# install binary
    install (TARGETS e57processor DESTINATION bin)
endif()

