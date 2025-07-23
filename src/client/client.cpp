#include <vector>

#include <spdlog/spdlog.h>

#include "client_lib.h"

int main() {
    Client client(7);

    Timer complete_search_timer;
    Timer get_query_timer;
    Timer get_centroids_timer;
    Timer sort_centroids_timer;
    Timer encrypt_query_subvector_lens_timer;

    SPDLOG_INFO("Starting query and timers");

    complete_search_timer.StartTimer();

    // Future TODO: Parallelise query retrieval while fetching and
    // computing nearest centroids
    // INFO: Encryption params are sent from the server, so query encryption
    // only after initial query to server

    get_query_timer.StartTimer();
    std::vector<float> precise_queries = client.get_query();
    get_query_timer.StopTimer();
    SPDLOG_INFO("Query vectors obtained successfully, time(microseconds) = {}",
                get_query_timer.getDurationMicroseconds());

    get_centroids_timer.StartTimer();
    auto [centroids, encrypted_parms] = client.get_centroids_encrypted_parms();
    get_centroids_timer.StopTimer();
    SPDLOG_INFO(
        "Fetched centroids from server successfully, time(microseconds) = {}",
        get_centroids_timer.getDurationMicroseconds());

    client.init_client_encrypt_parms(encrypted_parms);

    sort_centroids_timer.StartTimer();
    std::vector<faiss_idx_t> computed_nearest_centroids_idx =
        client.sort_nearest_centroids(precise_queries, centroids);
    sort_centroids_timer.StopTimer();
    SPDLOG_INFO(
        "Computed nearest centroids successfully, time(microseconds) = {}",
        sort_centroids_timer.getDurationMicroseconds());

    encrypt_query_subvector_lens_timer.StartTimer();
    auto [encrypted_subvectors, encrypted_subvectors_squared] =
        client.compute_encrypted_subvector_components(precise_queries);
    encrypt_query_subvector_lens_timer.StopTimer();
    SPDLOG_INFO("Computed encrypted subvector and squared lengths, "
                "time(microseconds) = {}",
                encrypt_query_subvector_lens_timer.getDurationMicroseconds());

    // Send nearest centroids to server to compute coarse scores (distances)
    std::vector<float> coarse_distance_scores;
    std::vector<faiss_idx_t> coarse_vector_indexes;
    std::vector<size_t> list_sizes_per_query_coarse;
    client.get_encrypted_coarse_scores(
        encrypted_subvectors, encrypted_subvectors_squared,
        coarse_distance_scores, coarse_vector_indexes,
        list_sizes_per_query_coarse);
    SPDLOG_INFO("Received coarse distance scores successfully");

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
