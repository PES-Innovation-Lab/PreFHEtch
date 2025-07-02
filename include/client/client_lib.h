#pragma once

#include <map>
#include <vector>

constexpr int64_t PRECISE_VECTOR_DIMENSIONS = 128;
constexpr int64_t COARSE_VECTOR_DIMENSIONS = 8;
constexpr int64_t NPROBE = 5;

const std::string server_addr = "http://localhost:8080/";

// Replace std::vector<float> with the corresponding Encrypted Vector type
void ping_server();

void get_query(std::array<float, PRECISE_VECTOR_DIMENSIONS> &query);
// In-progress (Quantisation)
void compute_coarse_query(
    std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    const std::vector<float> &quantisation_params,
    std::array<float, COARSE_VECTOR_DIMENSIONS> &coarse_query);

void get_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids);
void sort_nearest_centroids(
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    const std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids,
    std::map<float, int64_t> &nearest_centroids_idx);

// std::vector<float> compute_encrypted_coarse_query(const std::vector<float>
// &); std::vector<float> compute_encrypted_precise_query(const
// std::vector<float> &);

void get_coarse_scores(
    const std::array<int64_t, NPROBE> &nearest_centroids_idx,
    const std::array<float, COARSE_VECTOR_DIMENSIONS> &coarse_query,
    std::vector<float> &coarse_scores);
void compute_nearest_coarse_vectors(
    const std::vector<float> &coarse_scores,
    std::vector<int64_t> &nearest_coarse_vectors_idx);

void get_precise_scores(
    const std::vector<int64_t> &nearest_coarse_vectors_idx,
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::vector<float> &precise_scores);
void compute_nearest_precise_vectors(
    const std::vector<float> &precise_scores,
    std::vector<int64_t> &nearest_precise_vectors_idx);

void get_precise_vectors_pir(
    const std::vector<int64_t> &nearest_precise_vectors_idx,
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &query_results);
