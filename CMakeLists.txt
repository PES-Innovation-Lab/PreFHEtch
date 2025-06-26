cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

project(PreFHEtch
        VERSION 0.1
        LANGUAGES CXX
)

# Web framework to build servers
find_package(Drogon)
# Library to make HTTP requests
find_package(cpr CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.15.3
)

FetchContent_MakeAvailable(spdlog)

# FAISS and SEAL - Enable when required

#find_package(faiss)
#
#FetchContent_Declare(
#        SEAL
#        GIT_REPOSITORY https://github.com/microsoft/SEAL.git
#        GIT_TAG 7a931d55ba84a40b85938f6ca3ac206f18654093
#)
#
#FetchContent_MakeAvailable(SEAL)

# PreFHEtch_server

add_library(PreFHEtch_server_lib
        src/server/server_lib.cpp
        src/server/controllers/Query.h
        src/server/controllers/Query.cc
)

target_include_directories(PreFHEtch_server_lib
        PUBLIC
        "include/server"
        "{spdlog_SOURCE_DIR}/include"
)

target_link_libraries(PreFHEtch_server_lib
        PRIVATE
        Drogon::Drogon
        spdlog::spdlog
        #  SEAL::seal
        #  faiss
)

add_executable(PreFHEtch_server
        src/server/server.cpp
)

target_link_libraries(PreFHEtch_server
        PRIVATE
        PreFHEtch_server_lib
        spdlog::spdlog
)

# PreFHEtch_client

add_library(PreFHEtch_client_lib
        src/client/client_lib.cpp
)

target_include_directories(PreFHEtch_client_lib
        PUBLIC include/client
)

target_link_libraries(PreFHEtch_client_lib
        PRIVATE
        cpr::cpr
        spdlog::spdlog
)

add_executable(PreFHEtch_client
        src/client/client.cpp
)

target_link_libraries(PreFHEtch_client
        PRIVATE
        PreFHEtch_client_lib
        cpr::cpr
        spdlog::spdlog
)
