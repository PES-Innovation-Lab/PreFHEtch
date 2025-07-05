#include <vector>

#include <spdlog/spdlog.h>

#include "client_lib.h"

int main() {
    // ping_server();
    // Replace std::vector<float> with the corresponding Encrypted Vector type

    std::array<float, PRECISE_VECTOR_DIMENSIONS> precise_query;
    get_query(precise_query);
    SPDLOG_INFO("Query vector obtained successfully");

    // Query quantisation
    // Get parameters from server ( refactor to get_centroids_params()? )

    // // To be refactored
    // std::vector<float> quantisation_params;
    //
    // std::array<float, COARSE_VECTOR_DIMENSIONS> coarse_query;
    // compute_coarse_query(precise_query, quantisation_params, coarse_query);

    // Get centroids from server
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> centroids;
    get_centroids(centroids);
    SPDLOG_INFO("Fetched centroids from server successfully");

    // Compute nearest centroids
    std::vector<DistanceIndexData> nearest_centroids;
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
    std::array<size_t, NQUERY> list_sizes_per_query;
    get_coarse_scores(nearest_centroids, precise_query,
                      coarse_distance_scores, coarse_vector_indexes,
                      list_sizes_per_query);
    SPDLOG_INFO("Received coarse distance scores successfully");

    std::vector<DistanceIndexData> nearest_coarse_vectors;
    compute_nearest_coarse_vectors(coarse_distance_scores,
                                   coarse_vector_indexes, list_sizes_per_query,
                                   nearest_coarse_vectors);
    SPDLOG_INFO("Computed nearest coarse vectors successfully");

    // std::vector<float> precise_scores;
    // get_precise_scores(nearest_coarse_vectors_idx, precise_query,
    //                    precise_scores);
    // std::vector<faiss_idx_t> nearest_precise_vectors_idx;
    // compute_nearest_precise_vectors(precise_scores,
    //                                 nearest_precise_vectors_idx);
    //
    // // Get query vector results
    // std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> query_results;
    // get_precise_vectors_pir(nearest_precise_vectors_idx, query_results);

    return 0;
}
