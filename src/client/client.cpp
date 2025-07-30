#include <vector>

#include <boost/program_options.hpp>
#include <seal/seal.h>
#include <spdlog/spdlog.h>

#include "client_lib.h"
#include "client_server_utils.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    if (BFV_SCALING_FACTOR != 1) {
        SPDLOG_ERROR("BFV_SCALING_FACTOR = 1");
        throw std::runtime_error("BFV_SCALING_FACTOR != 1");
    }

    bool enable_single_phase = false;
    size_t num_queries, nprobe, coarse_probe, k_nearest;

    try {

        po::options_description desc("Allowed options");
        desc.add_options()("help", "help message")("nq", po::value<size_t>(),
                                                   "number of queries")(
            "nprobe", po::value<size_t>(),
            "number of nearest centroids to probe")(
            "coarse-probe", po::value<size_t>(),
            "nearest coarse vectors to perform a precise search")(
            "k", po::value<size_t>(), "required number of resultant vectors")(
            "single-phase", po::bool_switch(&enable_single_phase),
            "perform single-phase search");

        po::variables_map po_vm;
        po::store(po::parse_command_line(argc, argv, desc), po_vm);
        po::notify(po_vm);

        if (po_vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        num_queries = po_vm["nq"].as<size_t>();
        nprobe = po_vm["nprobe"].as<size_t>();
        coarse_probe = po_vm["coarse-probe"].as<size_t>();
        k_nearest = po_vm["k"].as<size_t>();

    } catch (std::exception &e) {
        SPDLOG_ERROR("Error while parsing command line args = {}", e.what());
        return 1;
    }

    Client client(num_queries, nprobe, coarse_probe, k_nearest);

    Timer complete_search_timer;
    Timer get_query_timer;
    Timer get_centroids_timer;
    Timer sort_centroids_timer;

    SPDLOG_INFO("Enable single phase search = {}", enable_single_phase);
    SPDLOG_INFO("Starting query and timers");
    SPDLOG_INFO("num_queries = {}, nprobe = {}, coarse_probe = {}", num_queries,
                nprobe, coarse_probe);

    // Future TODO: Parallelise query retrieval while fetching and
    // computing nearest centroids
    // INFO: Encryption params are sent from the server, so query encryption
    // only after initial query to server

    complete_search_timer.StartTimer();

    get_query_timer.StartTimer();
    std::vector<float> precise_queries = client.get_query();
    get_query_timer.StopTimer();
    SPDLOG_INFO("Query vectors obtained successfully, time = {}(us)",
                get_query_timer.getDurationMicroseconds());

    get_centroids_timer.StartTimer();
    auto [centroids, encrypted_parms] = client.get_centroids_encrypted_parms();
    get_centroids_timer.StopTimer();
    SPDLOG_INFO("Fetched centroids from server successfully, time = {}(us)",
                get_centroids_timer.getDurationMicroseconds());

    client.init_client_encrypt_parms(encrypted_parms);

    sort_centroids_timer.StartTimer();
    auto [sort_nearest_centroids_idx, nprobe_nearest_centroids_idx] =
        client.sort_nearest_centroids(precise_queries, centroids);
    sort_centroids_timer.StopTimer();
    SPDLOG_INFO("Computed nearest centroids successfully, time = {}(us)",
                sort_centroids_timer.getDurationMicroseconds());

    if (enable_single_phase) {

        Timer encrypt_search_params_timer;
        Timer search_timer;
        Timer deserialise_search_results_timer;
        Timer compute_k_nearest_precise_timer;

        encrypt_search_params_timer.StartTimer();
        auto [serde_encrypted_queries, serde_relin_keys, serde_galois_keys] =
            client.compute_encrypted_single_phase_search_parms(precise_queries);
        encrypt_search_params_timer.StopTimer();
        SPDLOG_INFO("Computed encrypted single phase search params "
                    "time = {}(us)",
                    encrypt_search_params_timer.getDurationMicroseconds());

        search_timer.StartTimer();
        auto [serde_encrypted_distances, result_vector_labels] =
            client.get_encrypted_single_phase_search_scores(
                serde_encrypted_queries, nprobe_nearest_centroids_idx,
                serde_relin_keys, serde_galois_keys);
        search_timer.StopTimer();
        SPDLOG_INFO(
            "Received encrypted single phase search distances successfully, "
            "time = {}(us)",
            search_timer.getDurationMicroseconds());

        deserialise_search_results_timer.StartTimer();
        auto decrypted_distances = client.deserialise_decrypt_coarse_distances(
            serde_encrypted_distances);
        deserialise_search_results_timer.StopTimer();
        SPDLOG_INFO("Deserialised and decrypted search distances successfully, "
                    "time = {}(us)",
                    deserialise_search_results_timer.getDurationMicroseconds());

        compute_k_nearest_precise_timer.StartTimer();
        std::vector<std::vector<faiss_idx_t>> k_nearest_vector_ids =
            client.compute_nearest_vectors_id(decrypted_distances,
                                              result_vector_labels, num_queries,
                                              k_nearest);
        compute_k_nearest_precise_timer.StopTimer();
        SPDLOG_INFO(
            "Computed nearest precise vectors successfully, time = {}(us)\n",
            compute_k_nearest_precise_timer.getDurationMicroseconds());

        complete_search_timer.StopTimer();
        SPDLOG_INFO("Complete Single Phase Search time = {}(us)",
                    complete_search_timer.getDurationMicroseconds());

        SPDLOG_INFO("Single Phase Search Query completed!");

        client.benchmark_results(k_nearest_vector_ids);

    } else {

        Timer encrypt_coarse_search_params_timer;
        Timer coarse_search_timer;
        Timer deserialise_coarse_search_results_timer;
        Timer compute_nearest_nprobe_coarse_search_timer;
        Timer precise_search_timer;
        Timer deserialise_precise_search_results_timer;
        Timer compute_k_nearest_precise_timer;

        encrypt_coarse_search_params_timer.StartTimer();
        auto [serde_encrypted_residual_queries,
              serde_encrypted_residual_queries_squared, serde_relin_keys,
              serde_galois_keys, serde_encrypted_precise_queries] =
            client.compute_encrypted_two_phase_search_parms(
                precise_queries, centroids, sort_nearest_centroids_idx);
        encrypt_coarse_search_params_timer.StopTimer();
        SPDLOG_INFO(
            "Computed encrypted coarse search params "
            "time = {}(us)",
            encrypt_coarse_search_params_timer.getDurationMicroseconds());

        coarse_search_timer.StartTimer();
        auto [serde_encrypted_coarse_distances, coarse_vector_labels] =
            client.get_encrypted_coarse_scores(
                serde_encrypted_residual_queries,
                serde_encrypted_residual_queries_squared,
                nprobe_nearest_centroids_idx, serde_relin_keys,
                serde_galois_keys);
        coarse_search_timer.StopTimer();
        SPDLOG_INFO("Received encrypted coarse distances successfully, "
                    "time = {}(us)",
                    coarse_search_timer.getDurationMicroseconds());

        deserialise_coarse_search_results_timer.StartTimer();
        auto decrypted_coarse_distances =
            client.deserialise_decrypt_coarse_distances(
                serde_encrypted_coarse_distances);
        deserialise_coarse_search_results_timer.StopTimer();
        SPDLOG_INFO(
            "Deserialised and decrypted coarse distances successfully, "
            "time = {}(us)",
            deserialise_coarse_search_results_timer.getDurationMicroseconds());

        compute_nearest_nprobe_coarse_search_timer.StartTimer();
        std::vector<std::vector<faiss_idx_t>> nearest_coarse_labels =
            client.compute_nearest_vectors_id(decrypted_coarse_distances,
                                              coarse_vector_labels, nprobe,
                                              coarse_probe);
        compute_nearest_nprobe_coarse_search_timer.StopTimer();
        SPDLOG_INFO(
            "Computed nearest coarse vectors successfully, time = {}(us)",
            compute_nearest_nprobe_coarse_search_timer
                .getDurationMicroseconds());

        precise_search_timer.StartTimer();
        std::vector<std::vector<std::vector<seal::seal_byte>>>
            serde_encrypted_precise_scores =
                client.get_precise_scores(serde_encrypted_precise_queries,
                                          nearest_coarse_labels,
                                          serde_relin_keys, serde_galois_keys);
        precise_search_timer.StopTimer();
        SPDLOG_INFO(
            "Received precise distance scores successfully, time = {}(us)",
            precise_search_timer.getDurationMicroseconds());

        deserialise_precise_search_results_timer.StartTimer();
        std::vector<std::vector<float>> decrypted_precise_distances =
            client.deserialise_decrypt_precise_distances(
                serde_encrypted_precise_scores);
        deserialise_precise_search_results_timer.StopTimer();
        SPDLOG_INFO(
            "Deserialised and decrypted precise distances successfully, "
            "time = {}(us)",
            deserialise_precise_search_results_timer.getDurationMicroseconds());

        compute_k_nearest_precise_timer.StartTimer();
        std::vector<std::vector<faiss_idx_t>> k_nearest_vector_ids =
            client.compute_nearest_vectors_id(decrypted_precise_distances,
                                              nearest_coarse_labels,
                                              num_queries, k_nearest);
        compute_k_nearest_precise_timer.StopTimer();
        SPDLOG_INFO(
            "Computed nearest precise vectors successfully, time = {}(us)\n",
            compute_k_nearest_precise_timer.getDurationMicroseconds());

        complete_search_timer.StopTimer();
        SPDLOG_INFO("Complete Two Phase Search time = {}(us)",
                    complete_search_timer.getDurationMicroseconds());

        SPDLOG_INFO("Two Phase Search Query completed!");

        client.benchmark_results(k_nearest_vector_ids);
    }

    // // Get query vector results
    // std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
    //            NQUERY>
    //     query_results;
    // std::array<std::array<faiss_idx_t, K>, NQUERY> query_results_idx;
    // get_precise_vectors_pir(nearest_precise_vectors, query_results,
    //                         query_results_idx);

    return 0;
}
