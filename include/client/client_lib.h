#pragma once

#include <vector>

#include "client_server_utils.h"

const std::string server_addr = "http://localhost:8080/";

struct DistanceIndexData {
    float distance;
    faiss_idx_t idx;
};

class Client {
  private:
    size_t m_PreciseVectorDimensions;
    size_t m_NumQueries;

  public:
    void set_num_queries(size_t num_queries);

    void get_query(std::vector<std::vector<float>> &queries);

    void get_centroids(std::vector<std::vector<float>> &centroids) const;
    void sort_nearest_centroids(
        const std::vector<std::vector<float>> &precise_queries,
        const std::vector<std::vector<float>> &centroids,
        std::vector<std::vector<faiss_idx_t>> &computed_nearest_centroids_idx)
        const;

    void get_coarse_scores(
        const std::array<std::vector<DistanceIndexData>, NQUERY>
            &sorted_centroids,
        // TODO: Sending precise query temporarily, will be sending coarse
        // vector in a future implementation
        const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
            &precise_query,
        std::vector<float> &coarse_scores,
        std::vector<faiss_idx_t> &coarse_vectors_idx,
        std::array<size_t, NQUERY> &list_sizes_per_query_coarse);

    void compute_nearest_coarse_vectors(
        const std::vector<float> &coarse_distance_scores,
        const std::vector<faiss_idx_t> &coarse_vector_indexes,
        const std::array<size_t, NQUERY> &list_sizes_per_query_coarse,
        std::array<std::vector<DistanceIndexData>, NQUERY>
            &nearest_coarse_vectors_idx);

    void get_precise_scores(
        const std::array<std::vector<DistanceIndexData>, NQUERY>
            &sorted_coarse_vectors,
        const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
            &precise_query,
        std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores);

    void compute_nearest_precise_vectors(
        const std::array<std::array<float, COARSE_PROBE>, NQUERY>
            &precise_scores,
        // Uses the same index order for precise scores
        const std::array<std::vector<DistanceIndexData>, NQUERY>
            &sorted_coarse_vectors,
        std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
            &nearest_precise_vectors);

    void get_precise_vectors_pir(
        const std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
            &nearest_precise_vectors,
        std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                   NQUERY> &query_results,
        std::array<std::array<faiss_idx_t, K>, NQUERY> &query_results_idx);

    void benchmark_results(const std::array<std::array<faiss_idx_t, K>, NQUERY>
                               &query_results_idx);
};
