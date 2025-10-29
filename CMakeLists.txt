set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 3.10)
project(mpi_hooks)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

set(COMMON_FLAGS "-Wall -Wextra -fPIC -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

add_library(sampiprofile SHARED sampiprofile.cc)
target_include_directories(sampiprofile PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(sampiprofile ${MPI_CXX_LIBRARIES})

add_library(sampiboost SHARED sampiboost.cc)
target_include_directories(sampiboost PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(sampiboost ${MPI_CXX_LIBRARIES})

set_target_properties(sampiprofile PROPERTIES OUTPUT_NAME "sampiprofile")
set_target_properties(sampiboost PROPERTIES OUTPUT_NAME "sampiboost")

install(TARGETS sampiprofile sampiboost
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
