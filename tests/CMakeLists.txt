cmake_minimum_required(VERSION 3.30)

project(benchmarks)

include(FetchContent)

FetchContent_Declare(
  faiss
  # GIT_REPOSITORY https://github.com/tejas-techstack/faiss.git
  # GIT_TAG       752b687150f1edf9a7b97f6acaed9f392d9d677f 
  GIT_REPOSITORY https://github.com/PES-Innovation-Lab/PreFHEtch-faiss.git 
  GIT_TAG        84c818c469c3c5d23e268753c984271d651cd62c
)

FetchContent_MakeAvailable(faiss)

add_executable(full_search full_search.cpp)

target_link_libraries(full_search PRIVATE faiss)
