cmake_minimum_required(VERSION 3.10)
project(sealauto VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(sealauto ecfs.cpp)

find_package(SEAL REQUIRED)

find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FAISS QUIET faiss)
endif()

if(NOT FAISS_FOUND)
    set(FAISS_SEARCH_PATHS
        /usr/local
        /usr
        /opt/faiss
        $ENV{FAISS_ROOT}
        ${CMAKE_PREFIX_PATH}
    )
    
    # Find FAISS include directory
    find_path(FAISS_INCLUDE_DIR
        NAMES faiss/IndexFlat.h faiss/IndexPQ.h
        PATHS ${FAISS_SEARCH_PATHS}
        PATH_SUFFIXES include
        DOC "FAISS include directory"
    )
    
    find_library(FAISS_LIBRARY
        NAMES faiss libfaiss
        PATHS ${FAISS_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        DOC "FAISS library"
    )
    
    if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
        set(FAISS_FOUND TRUE)
        set(FAISS_INCLUDE_DIRS ${FAISS_INCLUDE_DIR})
        set(FAISS_LIBRARIES ${FAISS_LIBRARY})
        message(STATUS "Found FAISS: ${FAISS_LIBRARY}")
        message(STATUS "FAISS include dir: ${FAISS_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "FAISS not found! Please install FAISS or set FAISS_ROOT environment variable.")
    endif()
endif()

target_link_libraries(sealauto 
    SEAL::seal 
    ${FAISS_LIBRARIES}
)

target_include_directories(sealauto PRIVATE 
    ${FAISS_INCLUDE_DIRS}
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sealauto OpenMP::OpenMP_CXX)
endif()

find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(sealauto ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

message(STATUS "SEAL found: ${SEAL_FOUND}")
message(STATUS "FAISS found: ${FAISS_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
endif()
if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_FOUND}")
endif()
if(LAPACK_FOUND)
    message(STATUS "LAPACK found: ${LAPACK_FOUND}")
endif()
