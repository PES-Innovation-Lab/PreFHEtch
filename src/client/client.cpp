#include <vector>

#include <spdlog/spdlog.h>

#include "client_lib.h"

int main() {
    Client client;
    client.set_num_queries(7);
    Timer precise_benchmark_timer;

    SPDLOG_INFO("Starting query");
    precise_benchmark_timer.StartTimer();

    std::vector<std::vector<float>> precise_queries;
    client.get_query(precise_queries);
    SPDLOG_INFO("Query vectors obtained successfully");

    // Get centroids from server
    std::vector<std::vector<float>> centroids;
    client.get_centroids(centroids);
    SPDLOG_INFO("Fetched centroids from server successfully");

    // Compute nearest centroids per query
    std::vector<std::vector<faiss_idx_t>> computed_nearest_centroids_idx;
    client.sort_nearest_centroids(precise_queries, centroids,
                                  computed_nearest_centroids_idx);
    SPDLOG_INFO("Computed nearest centroids successfully");

    // // Send nearest centroids to server to compute coarse scores (distances)
    // std::vector<float> coarse_distance_scores;
    // std::vector<faiss_idx_t> coarse_vector_indexes;
    // std::array<size_t, NQUERY> list_sizes_per_query_coarse;
    // get_coarse_scores(nearest_centroids, precise_query,
    // coarse_distance_scores,
    //                   coarse_vector_indexes, list_sizes_per_query_coarse);
    // // SPDLOG_INFO("Received coarse distance scores successfully");
    //
    // std::array<std::vector<DistanceIndexData>, NQUERY>
    // nearest_coarse_vectors; compute_nearest_coarse_vectors(
    //     coarse_distance_scores, coarse_vector_indexes,
    //     list_sizes_per_query_coarse, nearest_coarse_vectors);
    // // SPDLOG_INFO("Computed nearest coarse vectors successfully");
    //
    // // Send nearest coarse vector indexes to server to compute precise scores
    // // (distances)
    // std::array<std::array<float, COARSE_PROBE>, NQUERY> precise_scores;
    // get_precise_scores(nearest_coarse_vectors, precise_query,
    // precise_scores);
    // // SPDLOG_INFO("Received precise distance scores successfully");
    //
    // std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
    //     nearest_precise_vectors;
    // compute_nearest_precise_vectors(precise_scores, nearest_coarse_vectors,
    //                                 nearest_precise_vectors);
    // // SPDLOG_INFO("Computed nearest precise vectors successfully");
    //
    // precise_benchmark_timer.StopTimer();
    //
    // printf("\n");
    // SPDLOG_INFO("TIME");
    // SPDLOG_INFO("Start: Query, End: Computing nearest precise vectors (Does "
    //             "not include PIR)");
    //
    // long long time_micro = 0;
    // long long time_milli = 0;
    // precise_benchmark_timer.getDuration(time_micro, time_milli);
    // SPDLOG_INFO("Time in microseconds = {}, Time in milliseconds = {}",
    //             time_micro, time_milli);
    //
    // // Get query vector results
    // std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
    //            NQUERY>
    //     query_results;
    // std::array<std::array<faiss_idx_t, K>, NQUERY> query_results_idx;
    // get_precise_vectors_pir(nearest_precise_vectors, query_results,
    //                         query_results_idx);
    //
    // SPDLOG_INFO("Query completed!");
    // benchmark_results(query_results_idx);

    return 0;
}
