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

# FAISS and SEAL - Enable when required

find_package(faiss REQUIRED)

#FetchContent_Declare(
#        SEAL
#        GIT_REPOSITORY https://github.com/microsoft/SEAL.git
#        GIT_TAG 7a931d55ba84a40b85938f6ca3ac206f18654093
#)
#
#FetchContent_MakeAvailable(SEAL)

# PreFHEtch_server

add_library(PreFHEtch_server_headers INTERFACE)
target_include_directories(PreFHEtch_server_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/server)
target_link_libraries(PreFHEtch_server_headers
        INTERFACE
        nlohmann_json::nlohmann_json
        Drogon::Drogon
        spdlog::spdlog
        faiss
        #  SEAL::seal
)

add_library(PreFHEtch_server_lib
        src/server/server_lib.cpp
        src/server/controllers/Query.h
        src/server/controllers/Query.cc
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
target_include_directories(PreFHEtch_client_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/client)
target_link_libraries(PreFHEtch_client_headers
        INTERFACE
        nlohmann_json::nlohmann_json
        cpr::cpr
        spdlog::spdlog
        #  faiss
        #  SEAL::seal
)

add_library(PreFHEtch_client_lib
        src/client/client_lib.cpp
        src/client/client_utils.cpp
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
