#pragma once

#include <vector>

#include "client_server_utils.h"

const std::string server_addr = "http://localhost:8080/";

struct DistanceIndexData {
    float distance;
    faiss_idx_t idx;
};

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
    std::vector<DistanceIndexData> &nearest_centroids);

// std::vector<float> compute_encrypted_coarse_query(const std::vector<float>
// &); std::vector<float> compute_encrypted_precise_query(const
// std::vector<float> &);

void get_coarse_scores(
    std::vector<DistanceIndexData> &sorted_centroids,
    // Sending precise query temporarily, will be sending coarse vector in a
    // future implementation
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::vector<float> &coarse_scores,
    std::vector<faiss_idx_t> &coarse_vectors_idx,
    std::array<size_t, NQUERY> &list_sizes_per_query);

void compute_nearest_coarse_vectors(
    const std::vector<float> &coarse_distance_scores,
    const std::vector<faiss_idx_t> &coarse_vector_indexes,
    const std::array<size_t, NQUERY> &list_sizes_per_query,
    std::vector<DistanceIndexData> &nearest_coarse_vectors_idx);

void get_precise_scores(
    const std::vector<faiss_idx_t> &sorted_centroids,
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::vector<float> &precise_scores);
void compute_nearest_precise_vectors(
    const std::vector<float> &precise_scores,
    std::vector<faiss_idx_t> &nearest_precise_vectors_idx);

void get_precise_vectors_pir(
    const std::vector<faiss_idx_t> &nearest_precise_vectors_idx,
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &query_results);
