#include <algorithm>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

#include "client_server_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";
char const *GROUNDTRUTH_DATASET_PATH =
    "../sift/siftsmall/siftsmall_groundtruth.ivecs";

void Client::set_num_queries(size_t num_queries) { m_NumQueries = num_queries; }

void Client::get_query(std::vector<std::vector<float>> &precise_queries) {
    size_t parsed_num_queries;
    std::vector<float> parsed_precise_queries;

    vecs_read<float>(QUERY_DATASET_PATH, m_PreciseVectorDimensions,
                     parsed_num_queries, parsed_precise_queries);

    if (m_NumQueries > parsed_num_queries) {
        SPDLOG_ERROR("insufficient queries present in dataset");
        throw std::runtime_error("insufficient queries present in dataset");
    }

    precise_queries.reserve(m_NumQueries);

    for (int i = 0; i < m_NumQueries; i++) {
        std::vector<float> query;
        query.reserve(m_PreciseVectorDimensions);

        for (int j = 0; j < m_PreciseVectorDimensions; j++) {
            size_t idx = i * m_PreciseVectorDimensions + j;
            query.push_back(parsed_precise_queries[idx]);
        }

        precise_queries.push_back(query);
    }

    // SPDLOG_INFO("Printing parsed num_queries = {}", num_queries);
    // for (const auto &query : precise_queries) {
    //     printf("Next Query:");
    //     for (const auto &dim : query) {
    //         printf("%f, ", dim);
    //     }
    //     printf("\n");
    // }
}

void Client::get_centroids(std::vector<std::vector<float>> &centroids) const {
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));

    const nlohmann::json resp = nlohmann::json::parse(r.text);
    centroids = resp.at("centroids").get<std::vector<std::vector<float>>>();

    SPDLOG_INFO("Centroids = {}", resp.dump());
}

void Client::sort_nearest_centroids(
    const std::vector<std::vector<float>> &precise_queries,
    const std::vector<std::vector<float>> &centroids,
    std::vector<std::vector<faiss_idx_t>> &computed_nearest_centroids_idx)
    const {
    computed_nearest_centroids_idx.reserve(m_NumQueries);

    std::vector<std::vector<DistanceIndexData>> nquery_centroids_distance;
    nquery_centroids_distance.reserve(m_NumQueries);

    for (int i = 0; i < m_NumQueries; i++) {
        std::vector<DistanceIndexData> centroid_distances;
        centroid_distances.reserve(centroids.size());

        for (int j = 0; j < centroids.size(); j++) {
            float distance = 0.0;
            for (int k = 0; k < m_PreciseVectorDimensions; k++) {
                distance +=
                    std::pow(precise_queries[i][k] - centroids[j][k], 2);
            }
            centroid_distances.push_back(DistanceIndexData{
                distance,
                j,
            });
        }

        nquery_centroids_distance.push_back(centroid_distances);
    }

    for (std::vector<DistanceIndexData> &query : nquery_centroids_distance) {
        std::ranges::sort(
            query, [&](const DistanceIndexData &a, const DistanceIndexData &b) {
                return a.distance < b.distance;
            });
    }

    for (const auto &query : nquery_centroids_distance) {
        std::vector<faiss_idx_t> nearest_centroids;
        nearest_centroids.reserve(centroids.size());

        for (const DistanceIndexData &distance : query) {
            nearest_centroids.push_back(distance.idx);
        }

        computed_nearest_centroids_idx.push_back(nearest_centroids);
    }

    // for (const auto &query : nquery_centroids_distance) {
    //     SPDLOG_INFO("Next Query");
    //     for (const auto &centroid_distance : query) {
    //         SPDLOG_INFO("distance = {}, centroid = {}",
    //                     centroid_distance.distance, centroid_distance.idx);
    //     }
    // }
}

void get_coarse_scores(
    const std::array<std::vector<DistanceIndexData>, NQUERY> &sorted_centroids,
    // Sending precise query temporarily, will be sending coarse vector in a
    // future implementation
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    std::vector<float> &coarse_scores,
    std::vector<faiss_idx_t> &coarse_vectors_idx,
    std::array<size_t, NQUERY> &list_sizes_per_query) {

    std::array<std::array<faiss_idx_t, NPROBE>, NQUERY> nearest_centroids_id;

    for (int i = 0; i < NQUERY; i++) {
        if (NPROBE > sorted_centroids[i].size()) {
            SPDLOG_ERROR("Centroids count is not equal to NPROBE");
            throw std::runtime_error("Centroids count is not equal to NPROBE");
        }
        for (int j = 0; j < NPROBE; j++) {
            nearest_centroids_id[i][j] = sorted_centroids[i][j].idx;
        }
    }

    nlohmann::json coarse_search_json;
    coarse_search_json["preciseQuery"] = precise_query;
    coarse_search_json["nearestCentroidIndexes"] = nearest_centroids_id;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);
    // SPDLOG_INFO("Response = {}", resp.dump());

    coarse_scores = resp.at("coarseDistanceScores").get<std::vector<float>>();
    coarse_vectors_idx =
        resp.at("coarseVectorIndexes").get<std::vector<faiss_idx_t>>();
    list_sizes_per_query =
        resp.at("listSizesPerQuery").get<std::array<size_t, NQUERY>>();
}

void compute_nearest_coarse_vectors(
    const std::vector<float> &coarse_distance_scores,
    const std::vector<faiss_idx_t> &coarse_vector_indexes,
    const std::array<size_t, NQUERY> &list_sizes_per_query,
    std::array<std::vector<DistanceIndexData>, NQUERY>
        &nearest_coarse_vectors) {

    size_t current_vector_index = 0;

    for (int i = 0; i < NQUERY; i++) {
        if (list_sizes_per_query[i] < COARSE_PROBE) {
            SPDLOG_ERROR("Number of computed coarse scores is lesser than "
                         "COARSE_PROBE");
            throw std::runtime_error("Number of computed coarse scores is "
                                     "lesser than COARSE_PROBE");
        }
        nearest_coarse_vectors[i].reserve(list_sizes_per_query[i]);

        for (int j = 0; j < list_sizes_per_query[i]; j++) {
            const size_t idx = current_vector_index + j;
            nearest_coarse_vectors[i].push_back(DistanceIndexData{
                coarse_distance_scores[idx],
                coarse_vector_indexes[idx],
            });
        }
        current_vector_index += list_sizes_per_query[i];
    }

    for (std::vector<DistanceIndexData> &query : nearest_coarse_vectors) {
        std::ranges::sort(
            query, [&](const DistanceIndexData &a, const DistanceIndexData &b) {
                return a.distance < b.distance;
            });
    }
}

void get_precise_scores(
    const std::array<std::vector<DistanceIndexData>, NQUERY>
        &sorted_coarse_vectors,
    const std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, NQUERY>
        &precise_query,
    std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores) {

    std::array<std::array<faiss_idx_t, COARSE_PROBE>, NQUERY>
        nearest_coarse_vectors_id;

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < COARSE_PROBE; j++) {
            nearest_coarse_vectors_id[i][j] = sorted_coarse_vectors[i][j].idx;
        }
    }

    nlohmann::json coarse_search_json;
    coarse_search_json["preciseQuery"] = precise_query;
    coarse_search_json["nearestCoarseVectorIndexes"] =
        nearest_coarse_vectors_id;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "precisesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);

    precise_scores =
        resp.at("preciseDistanceScores")
            .get<std::array<std::array<float, COARSE_PROBE>, NQUERY>>();
}

void compute_nearest_precise_vectors(
    const std::array<std::array<float, COARSE_PROBE>, NQUERY> &precise_scores,
    const std::array<std::vector<DistanceIndexData>, NQUERY>
        &sorted_coarse_vectors,
    std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
        &nearest_precise_vectors) {
    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < COARSE_PROBE; j++) {
            nearest_precise_vectors[i][j] = DistanceIndexData{
                precise_scores[i][j], sorted_coarse_vectors[i][j].idx};
        }
    }

    for (auto &precise_score_query : nearest_precise_vectors) {
        std::ranges::sort(precise_score_query, [&](const DistanceIndexData &a,
                                                   const DistanceIndexData &b) {
            return a.distance < b.distance;
        });
    }
}

void get_precise_vectors_pir(
    const std::array<std::array<DistanceIndexData, COARSE_PROBE>, NQUERY>
        &nearest_precise_vectors,
    std::array<std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
               NQUERY> &query_results,
    std::array<std::array<faiss_idx_t, K>, NQUERY> &query_results_idx) {

    if constexpr (K > COARSE_PROBE) {
        SPDLOG_ERROR("K greater than COARSE_PROBE");
        throw std::runtime_error("K greater than COARSE_PROBE");
    }

    for (int i = 0; i < NQUERY; i++) {
        for (int j = 0; j < K; j++) {
            query_results_idx[i][j] = nearest_precise_vectors[i][j].idx;
        }
    }

    nlohmann::json precise_vector_pir_json;
    precise_vector_pir_json["nearestPreciseVectorIndexes"] = query_results_idx;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "precise-vector-pir"),
                                cpr::Body(precise_vector_pir_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);

    query_results =
        resp.at("queryResults")
            .get<std::array<
                std::array<std::array<float, PRECISE_VECTOR_DIMENSIONS>, K>,
                NQUERY>>();
}

void benchmark_results(const std::array<std::array<faiss_idx_t, K>, NQUERY>
                           &observed_query_results_idx) {

    SPDLOG_INFO("BENCHMARK RESULTS");

    size_t gt_nn_per_query;
    size_t gt_nq;
    std::vector<int> ground_truth;
    vecs_read(GROUNDTRUTH_DATASET_PATH, gt_nn_per_query, gt_nq, ground_truth);

    // MRR considers only the position of the first result
    // MRR@10 - Results outside top 10 are irrelevant
    float mrr_1 = 0, mrr_10 = 0, mrr_100 = 0;

    // Recall considers ratio of true results to ground truth
    int nq_recall_1 = 0, nq_recall_10 = 0, nq_recall_100 = 0;

    if (K > gt_nn_per_query) {
        SPDLOG_ERROR("K greater than nearest neigbours per query in ground "
                     "truth dataset");
        throw std::runtime_error(
            "K greater than nearest neigbours per query in ground "
            "truth dataset");
    }
    for (int i = 0; i < NQUERY; i++) {
        // printf("\n\n");
        // SPDLOG_INFO("QUERY BENCHMARKS Q = {}", i + 1);
        // SPDLOG_INFO("Ground truth nearest neighbours for Q = {}", i + 1);
        int recall_1 = 0, recall_10 = 0, recall_100 = 0;
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < K; k++) {
                if (ground_truth[i * gt_nn_per_query + j] ==
                    observed_query_results_idx[i][k]) {
                    if (k < 1)
                        recall_1++;
                    if (k < 10)
                        recall_10++;
                    if (k < 100)
                        recall_100++;

                    // Considering only 1st ground truth for MRR
                    if (j == 0) {
                        if (k < 1)
                            mrr_1 += 1.0f / static_cast<float>(k + 1);
                        if (k < 10)
                            mrr_10 += 1.0f / static_cast<float>(k + 1);
                        if (k < 100)
                            mrr_100 += 1.0f / static_cast<float>(k + 1);

                        // SPDLOG_INFO("Updated mrr = {}, {}, {}", mrr_1,
                        // mrr_10,
                        //             mrr_100);
                    }
                    break;
                }
            }
            // printf("%d, ", ground_truth[i * gt_nn_per_query + j]);
        }
        // printf("\n");

        // SPDLOG_INFO("Query Results:");
        // for (int j = 0; j < K; j++) {
        //     printf("%lld, ", observed_query_results_idx[i][j]);
        // }
        // printf("\n");

        // SPDLOG_INFO("Recall@1 = {}, Recall@10 = {}, Recall@100 = {}",
        //             static_cast<float>(recall_1) / 1,
        //             static_cast<float>(recall_10) / 10,
        //             static_cast<float>(recall_100) / 100);
        nq_recall_1 += recall_1;
        nq_recall_10 += recall_10;
        nq_recall_100 += recall_100;
    }

    printf("\n\n");
    SPDLOG_INFO("Total Query Benchmark Results");
    SPDLOG_INFO("Parameters: NPROBE = {}, COARSE_PROBE = {}, K = {}", NPROBE,
                COARSE_PROBE, K);
    SPDLOG_INFO("Parameters: NQUERY = {}, NLIST = {}", NQUERY, NLIST);
    SPDLOG_INFO("Parameters: SUB_QUANTIZERS = {}, SUB_VECTOR_SIZE = {}",
                SUB_QUANTIZERS, SUB_QUANTIZER_SIZE);
    SPDLOG_INFO("Recall@1 = {}, Recall@10 = {}, Recall@100 = {}",
                static_cast<float>(nq_recall_1) / (1 * NQUERY),
                static_cast<float>(nq_recall_10) / (10 * NQUERY),
                static_cast<float>(nq_recall_100) / (100 * NQUERY));
    SPDLOG_INFO("MRR@1 = {}, MRR@10 = {}, MRR@100 = {}", (float)mrr_1 / NQUERY,
                (float)mrr_10 / NQUERY, (float)mrr_100 / NQUERY);

    printf("\n\n");
    for (int i = 0; i < 100; i++) {
        printf("-");
    }
    printf("\n");
}
