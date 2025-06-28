#pragma once

#include <vector>

constexpr int64_t PRECISE_VECTOR_DIMENSIONS = 10;
constexpr int64_t COARSE_VECTOR_DIMENSIONS = 5;
constexpr int64_t NPROBE = 5;

constexpr int64_t NUM_CENTROIDS = 5;

void init_logger();
void run_server();

void retrieve_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids);