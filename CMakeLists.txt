cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

project(PreFHEtch
  VERSION 0.1
  LANGUAGES CXX
)

find_package(faiss)

include(FetchContent)

FetchContent_Declare(
  SEAL
  GIT_REPOSITORY https://github.com/microsoft/SEAL.git
  GIT_TAG 7a931d55ba84a40b85938f6ca3ac206f18654093
)

FetchContent_MakeAvailable(SEAL)

add_library(PreFHEtch_server_lib
  src/server/server_lib.cpp
)

target_include_directories(PreFHEtch_server_lib
  PUBLIC
  include/server
)

target_link_libraries(PreFHEtch_server_lib
  SEAL::seal
  faiss
)

add_executable(PreFHEtch_server
  src/server/server.cpp
)

target_link_libraries(PreFHEtch_server
  PreFHEtch_server_lib
)

add_library(PreFHEtch_client_lib
  src/client/client_lib.cpp
)

target_include_directories(PreFHEtch_client_lib
  PUBLIC
  include/client
)

add_executable(PreFHEtch_client
  src/client/client.cpp
)

target_link_libraries(PreFHEtch_client
  PreFHEtch_client_lib
)
