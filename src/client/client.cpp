#include <vector>

#include <spdlog/spdlog.h>

#include "client_lib.h"

int main() {
    // ping_server();
    // Replace std::vector<float> with the corresponding Encrypted Vector type

    std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        precise_query;
    get_query(precise_query);
    SPDLOG_INFO("Query vector obtained successfully");

    // Get centroids from server
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> centroids;
    get_centroids(centroids);
    SPDLOG_INFO("Fetched centroids from server successfully");

    // Compute nearest centroids
    std::array<std::vector<DistanceIndexData>, NQUERY> nearest_centroids;
    sort_nearest_centroids(precise_query, centroids, nearest_centroids);
    SPDLOG_INFO("Computed nearest centroids successfully");

    // // To be parallelised (async?)
    // const std::vector<float> encrypted_coarse_query =
    //     compute_encrypted_coarse_query(coarse_query);
    // // To be parallelised (async?)
    // const std::vector<float> encrypted_precise_query =
    //     compute_encrypted_precise_query(query);

    // Send nearest centroids to server to compute coarse scores (distances)
    std::vector<float> coarse_distance_scores;
    std::vector<faiss_idx_t> coarse_vector_indexes;
    std::array<size_t, NQUERY> list_sizes_per_query_coarse;
    get_coarse_scores(nearest_centroids, precise_query, coarse_distance_scores,
                      coarse_vector_indexes, list_sizes_per_query_coarse);
    SPDLOG_INFO("Received coarse distance scores successfully");

    std::array<std::vector<DistanceIndexData>, NQUERY> nearest_coarse_vectors;
    compute_nearest_coarse_vectors(
        coarse_distance_scores, coarse_vector_indexes,
        list_sizes_per_query_coarse, nearest_coarse_vectors);
    SPDLOG_INFO("Computed nearest coarse vectors successfully");

    // Send nearest coarse vector indexes to server to compute precise scores
    // (distances)
    std::array<std::array<float, COARSE_PROBE>, NQUERY> precise_scores;
    get_precise_scores(nearest_coarse_vectors, precise_query, precise_scores);
    SPDLOG_INFO("Received precise distance scores successfully");

    std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
        nearest_precise_vectors;
    compute_nearest_precise_vectors(precise_scores, nearest_coarse_vectors,
                                    nearest_precise_vectors);
    SPDLOG_INFO("Computed nearest precise vectors successfully");

    // Get query vector results
    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY>
        query_results;
    get_precise_vectors_pir(nearest_precise_vectors, query_results);

    benchmark_results(query_results);

    return 0;
}
