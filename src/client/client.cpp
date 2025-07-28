#include <vector>

#include <spdlog/spdlog.h>

#include "client_lib.h"

int main() {
    size_t num_queries = 1, nprobe = 1, coarse_probe = 300;
    Client client(num_queries, nprobe);

    Timer complete_search_timer;
    Timer get_query_timer;
    Timer get_centroids_timer;
    Timer sort_centroids_timer;
    Timer encrypt_coarse_search_params_timer;
    Timer coarse_search_timer;
    Timer deserialise_coarse_search_results_timer;
    Timer compute_nearest_nprobe_coarse_search_timer;

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
    auto [sort_nearest_centroids_idx, nprobe_nearest_centroids_idx] =
        client.sort_nearest_centroids(precise_queries, centroids);
    sort_centroids_timer.StopTimer();
    SPDLOG_INFO(
        "Computed nearest centroids successfully, time(microseconds) = {}",
        sort_centroids_timer.getDurationMicroseconds());

    encrypt_coarse_search_params_timer.StartTimer();
    auto [encrypted_subvectors, encrypted_subvectors_squared, serde_relin_keys,
          serde_galois_keys] =
        client.compute_encrypted_coarse_search_parms(
            precise_queries, centroids, sort_nearest_centroids_idx);
    encrypt_coarse_search_params_timer.StopTimer();
    SPDLOG_INFO("Computed encrypted subvector and squared lengths, "
                "time(microseconds) = {}",
                encrypt_coarse_search_params_timer.getDurationMicroseconds());

    coarse_search_timer.StartTimer();
    auto [serde_encrypted_coarse_distances, coarse_vector_labels] =
        client.get_encrypted_coarse_scores(
            encrypted_subvectors, encrypted_subvectors_squared,
            nprobe_nearest_centroids_idx, serde_relin_keys, serde_galois_keys);
    coarse_search_timer.StopTimer();
    SPDLOG_INFO("Received encrypted coarse distances successfully, "
                "time(microseconds) = {}",
                coarse_search_timer.getDurationMicroseconds());

    deserialise_coarse_search_results_timer.StartTimer();
    auto decrypted_coarse_distances =
        client.deserialise_decrypt_coarse_distances(
            serde_encrypted_coarse_distances);
    deserialise_coarse_search_results_timer.StopTimer();
    SPDLOG_INFO(
        "Deserialised and decrypted coarse distances successfully, "
        "time(microseconds) = {}",
        deserialise_coarse_search_results_timer.getDurationMicroseconds());

    compute_nearest_nprobe_coarse_search_timer.StartTimer();
    auto sorted_coarse_labels = client.compute_nearest_coarse_vectors_idx(
        decrypted_coarse_distances, coarse_vector_labels, nprobe, coarse_probe);
    compute_nearest_nprobe_coarse_search_timer.StopTimer();
    SPDLOG_INFO(
        "Computed nearest coarse vectors successfully, time(microseconds) = {}",
        compute_nearest_nprobe_coarse_search_timer.getDurationMicroseconds());

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
