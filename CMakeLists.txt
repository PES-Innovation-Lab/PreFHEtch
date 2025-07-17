cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(PreFHEtch
        VERSION 0.1
        LANGUAGES CXX
)

# Drogon JSON dependency
find_package(jsoncpp CONFIG REQUIRED)
# Web framework to build servers
find_package(Drogon REQUIRED)
# Library to make HTTP requests
find_package(cpr CONFIG REQUIRED)

include(FetchContent)

# Faiss fork to incorporate custom features
FetchContent_Declare(
        prefhetch-faiss
        GIT_REPOSITORY https://github.com/PES-Innovation-Lab/PreFHEtch-faiss
        GIT_TAG  54d04fa175b813f1edaa8d9282caf18c9b299425
)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Python bindings")
set(BUILD_TESTING OFF CACHE BOOL "Tests")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
FetchContent_MakeAvailable(prefhetch-faiss)

# JSON Library
FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.15.3
)
FetchContent_MakeAvailable(spdlog)

# PreFHEtch_server

add_library(PreFHEtch_server_headers INTERFACE)
target_include_directories(PreFHEtch_server_headers
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/server
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)
target_link_libraries(PreFHEtch_server_headers
        INTERFACE
        nlohmann_json::nlohmann_json
        Drogon::Drogon
        spdlog::spdlog
        faiss
)

add_library(PreFHEtch_server_lib
        src/server/server_lib.cpp
        src/server/server_utils.cpp
        src/server/controllers/Query.h
        src/server/controllers/Query.cc
        src/common/client_server_utils.cpp
)
target_link_libraries(PreFHEtch_server_lib
        PUBLIC
        PreFHEtch_server_headers
)

add_executable(PreFHEtch_server
        src/server/server.cpp
)
target_link_libraries(PreFHEtch_server
        PRIVATE
        PreFHEtch_server_lib
)

# PreFHEtch_client

add_library(PreFHEtch_client_headers INTERFACE)
target_include_directories(PreFHEtch_client_headers
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/client
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)
target_link_libraries(PreFHEtch_client_headers
        INTERFACE
        nlohmann_json::nlohmann_json
        cpr::cpr
        spdlog::spdlog
)

add_library(PreFHEtch_client_lib
        src/client/client_lib.cpp
        src/client/client_utils.cpp
        src/common/client_server_utils.cpp
)
target_link_libraries(PreFHEtch_client_lib
        PUBLIC
        PreFHEtch_client_headers
)

add_executable(PreFHEtch_client
        src/client/client.cpp
)
target_link_libraries(PreFHEtch_client
        PRIVATE
        PreFHEtch_client_lib
)
