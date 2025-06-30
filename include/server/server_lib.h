#pragma once

#include <vector>

#include <faiss/index_factory.h>

// Dataset - SIFT10K

constexpr int64_t PRECISE_VECTOR_DIMENSIONS = 128;
constexpr int64_t COARSE_VECTOR_DIMENSIONS = 5;

constexpr int64_t NUM_CENTROIDS = 5;
constexpr int64_t NPROBE = 5;
constexpr int64_t K = 100;

void init_logger();
void run_webserver();

void init_index();
void retrieve_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids);

void fvecs_read(const char *fname, size_t &d_out, size_t &n_out,
                std::vector<float> &vecs);
void ivecs_read(const char *fname, size_t &d_out, size_t &n_out,
                std::vector<int> &vecs);