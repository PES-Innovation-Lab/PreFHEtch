#include <algorithm>
#include <vector>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "client_lib.h"

#include "client_server_utils.h"

char const *QUERY_DATASET_PATH = "../sift/siftsmall/siftsmall_query.fvecs";

// Test method to ping server
void ping_server() {
    SPDLOG_INFO("Sending a request to /ping at {}", server_addr);
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "ping"));
    const nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}, Status code = {}", resp.dump(), r.status_code);
}

void get_query(std::array<float, PRECISE_VECTOR_DIMENSIONS> &query) {
    size_t nq;
    std::vector<float> xq;

    SPDLOG_INFO("Loading query:");

    size_t d2;
    vecs_read<float>(QUERY_DATASET_PATH, d2, nq, xq);
    assert(PRECISE_VECTOR_DIMENSIONS == d2 ||
           !"query does not have same dimension as train set");

    for (int i = 0; i < PRECISE_VECTOR_DIMENSIONS; i++) {
        query[i] = xq[i];
        printf("%.1f, ", xq[i]);
    }
    printf("\n");
}

void get_centroids(
    std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids) {
    SPDLOG_INFO("Sending a request to /query at {}", server_addr);
    cpr::Response r = cpr::Get(cpr::Url(server_addr + "query"));
    // SPDLOG_INFO("Response = {}, Status code = {}", r.text, r.status_code);

    const nlohmann::json resp = nlohmann::json::parse(r.text);
    centroids =
        resp.get<std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>>>();
    // SPDLOG_INFO("Retrieved centroids -> {}", resp.dump());
}

void sort_nearest_centroids(
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    const std::vector<std::array<float, PRECISE_VECTOR_DIMENSIONS>> &centroids,
    std::vector<DistanceIndexData> &nearest_centroids) {

    nearest_centroids.reserve(centroids.size());
    for (int i = 0; i < centroids.size(); i++) {
        float distance = 0.0;
        for (int j = 0; j < PRECISE_VECTOR_DIMENSIONS; j++) {
            distance += std::pow(precise_query[j] - centroids[i][j], 2);
        }
        nearest_centroids.push_back(DistanceIndexData{
            distance,
            i,
        });
    }

    std::sort(nearest_centroids.begin(), nearest_centroids.end(),
              [&](const DistanceIndexData &a, const DistanceIndexData &b) {
                  return a.distance < b.distance;
              });

    // for (const auto &vec: nearest_centroids) {
    //     SPDLOG_INFO("distance = {}, centroid = {}", vec.distance,
    //     vec.nearest_centroid_idx);
    // }
}

void get_coarse_scores(
    std::vector<DistanceIndexData> &sorted_centroids,
    // Sending precise query temporarily, will be sending coarse vector in a
    // future implementation
    const std::array<float, PRECISE_VECTOR_DIMENSIONS> &precise_query,
    std::vector<float> &coarse_scores,
    std::vector<faiss_idx_t> &coarse_vectors_idx,
    std::array<size_t, NQUERY> &list_sizes_per_query) {
    SPDLOG_INFO("Sending a request to /coarsesearch at {}", server_addr);

    std::array<faiss_idx_t, NPROBE> nearest_centroids_id;
    const size_t centroids_count = sorted_centroids.size();

    for (int i = 0; i < NPROBE; i++) {
        if (i >= centroids_count) {
            throw std::runtime_error("Centroids count is lesser than NPROBE");
        }
        nearest_centroids_id[i] = sorted_centroids[i].idx;
    }

    nlohmann::json coarse_search_json;
    coarse_search_json["preciseQuery"] = precise_query;
    coarse_search_json["nearestCentroidIndexes"] = nearest_centroids_id;

    cpr::Response r = cpr::Post(cpr::Url(server_addr + "coarsesearch"),
                                cpr::Body(coarse_search_json.dump()));

    nlohmann::json resp = nlohmann::json::parse(r.text);
    SPDLOG_INFO("Response = {}", resp.dump());

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
    std::vector<DistanceIndexData> &nearest_coarse_vectors) {
    size_t current_vector_index = 0;
    nearest_coarse_vectors.reserve(coarse_vector_indexes.size());

    for (int i = 0; i < list_sizes_per_query.size(); i++) {
        for (int j = 0; j < list_sizes_per_query[i]; j++) {
            const size_t idx = current_vector_index + j;
            nearest_coarse_vectors.push_back(DistanceIndexData{
                coarse_distance_scores[idx],
                coarse_vector_indexes[idx],
            });
        }
        current_vector_index += list_sizes_per_query[i];
    }

    std::sort(nearest_coarse_vectors.begin(), nearest_coarse_vectors.end(), [&](const DistanceIndexData &a, const DistanceIndexData &b) {
        return a.distance < b.distance;
    });

    // for (const DistanceIndexData &dist_ind_data : nearest_coarse_vectors) {
    //     SPDLOG_INFO("Distance  = {}, Coarse vector index = {}", dist_ind_data.distance,
    //                 dist_ind_data.idx);
    // }
}
